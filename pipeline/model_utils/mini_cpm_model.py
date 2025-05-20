import torch
import functools
from torch import Tensor
from transformers import AutoTokenizer, AutoProcessor
from typing import List, Optional, Tuple, Literal
from jaxtyping import Int, Float
from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.utils import get_orthogonalized_matrix
from prismatic.models.vlms.prismatic import PrismaticVLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.v2
from pipeline.utils.hook_utils import add_hooks

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


CPM_REFUSAL_TOKS  = [40, 2121]  # Example: 'I' 'As (check if needed)


def tokenize_instructions_minicpm(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace=True,
    pixel_values=None,
    model=None,
):  
    system_prompt=''
    images = []
    instructions_processed = []
    for image in pixel_values:
        images.append([image])
    for instruction in instructions:
        msgs_list = []
        if pixel_values is not None:
            content_parts = ["(<image>./</image>)", instruction]
        else:
            content_parts = [instruction]
        user_msg = {"role": "user", "content": "\n".join(content_parts)}

        # System prompt (if any)
        full_msg = []
        if system_prompt:
            full_msg.append({"role": "system", "content": system_prompt})
        full_msg.append(user_msg)

        msgs_list.append(full_msg)
        prompts_str = [
            model.processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in msgs_list
        ]
        instructions_processed.append(prompts_str[0])
    if pixel_values is not None:
        input_by_model = model.processor(instructions_processed, images, do_pad=True)
        input_by_model.pop("image_sizes")
    else:
        input_by_model = model.processor(instructions_processed, None, do_pad=True)
    return input_by_model, None

def orthogonalize_minicpmv_weights(basemodel, direction: torch.Tensor):
    lm = basemodel.model.llm.model  # ✅ Points to Qwen2Model

    # Embedding
    lm.embed_tokens.weight.data = get_orthogonalized_matrix(
        lm.embed_tokens.weight.data, direction
    )

    # Decoder layers: attention output + MLP down projection
    for block in lm.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T

        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.down_proj.weight.data.T, direction
        ).T

    # Optional: vision-language projector (resampler)
    if hasattr(basemodel.model, "resampler") and isinstance(basemodel.model.resampler.kv_proj, torch.nn.Linear):
        basemodel.model.resampler.kv_proj.weight.data = get_orthogonalized_matrix(
            basemodel.model.resampler.kv_proj.weight.data.T, direction
        ).T

def act_add_minicpmv_weights(basemodel, direction: torch.Tensor, coeff: float, layer: int):
    lm_layer = basemodel.model.llm.model.layers[layer - 1]

    dtype = lm_layer.mlp.down_proj.weight.dtype
    device = lm_layer.mlp.down_proj.weight.device
    bias = (coeff * direction).to(dtype=dtype, device=device)

    # Modify MLP down projection bias
    lm_layer.mlp.down_proj.bias = torch.nn.Parameter(bias)

    # Optional: modify vision-language projector bias
    if hasattr(basemodel.model, "resampler") and isinstance(basemodel.model.resampler.kv_proj, torch.nn.Linear):
        basemodel.model.resampler.kv_proj.bias = torch.nn.Parameter(bias)



def build_conversation_input_ids(
    tokenizer: "PreTrainedTokenizer",
    *,
    queries: List[str],
    answers: Optional[List[str]] = None,
    history: Optional[List[Tuple[str, str]]] = None,
    template_version: Optional[Literal["base", "chat", "vqa"]] = None,
    model=None,
):
    image_size: int = model.config.vision_config['image_size']
    patch_size: int = model.config.vision_config['patch_size']
    template_version = template_version or model.config.template_version
    # Token counts for vision tokens (from model arch spec)
    vision_token_num = (image_size // patch_size // 2) * (image_size // patch_size // 2) + 2

    history = history or []

    input_ids_list = []
    token_type_ids_list = []
    attention_masks = []
    labels_list = []

    for i, query in enumerate(queries):
        text = _history_to_prompt(template_version, history, query)
        input_ids = [tokenizer.bos_token_id] + [tokenizer.pad_token_id] * vision_token_num
        token_type_ids = [LANGUAGE_TOKEN_TYPE] + [VISION_TOKEN_TYPE] * vision_token_num

        text_ids = tokenizer.encode(text, add_special_tokens=False)

        if answers is not None and i < len(answers):
            answer_ids = tokenizer.encode(answers[i], add_special_tokens=False) + [tokenizer.eos_token_id]
            full_ids = text_ids + answer_ids
        else:
            answer_ids = None
            full_ids = text_ids

        input_ids += full_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(full_ids)

        attention_mask = [1] * len(input_ids)

        if answer_ids is not None:
            labels = [-100] * (len(input_ids) - len(answer_ids)) + answer_ids
        else:
            labels = None

        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids, dtype=torch.long))
        attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
        if labels is not None:
            labels_list.append(torch.tensor(labels, dtype=torch.long))

    # Pad all sequences to the max length (left-padding)
    def pad_to_max(seq_list, pad_value):
        max_len = max(seq.size(0) for seq in seq_list)
        padded = [torch.cat([torch.full((max_len - seq.size(0),), pad_value, dtype=seq.dtype), seq]) for seq in seq_list]
        return torch.stack(padded)

    input_ids = pad_to_max(input_ids_list, tokenizer.pad_token_id)
    token_type_ids = pad_to_max(token_type_ids_list, LANGUAGE_TOKEN_TYPE)
    attention_mask = pad_to_max(attention_masks, 0)
    labels = pad_to_max(labels_list, -100) if labels_list else None

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

def _history_to_prompt(signal_type, history, query):
    if signal_type == 'base':
        return query
    elif signal_type == 'vqa':
        answer_format = 'Short answer:'
    elif signal_type == 'chat':
        answer_format = 'Answer:'
    else:
        assert False, f"Unknown signal type {signal_type}"

    prompt = ''
    for i, (old_query, response) in enumerate(history):
        prompt += 'Question: ' + old_query + " {} ".format(answer_format) + response + "\n"
    prompt += 'Question: {} {}'.format(query, answer_format)
    return prompt

def transform_image(image):

    width, height = image.size
    max_dim = max(width, height)
    pad_width = (max_dim - width) // 2
    pad_height = (max_dim - height) // 2
    image_size = 1344

    transform = transforms.Compose(
        [
            torchvision.transforms.v2.Pad(
                (pad_width, pad_height, pad_width, pad_height), fill=0
            ),
            transforms.Resize(
                (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    image: torch.Tensor = transform(image) 
    return image

class MiniCPMV26(ModelBase):
    def __init__(self, model_path):
        super().__init__(model_path)
        
    # def __init__(self, model_name_or_path: str):
    #     self.model_name_or_path = model_name_or_path
    #     self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
    #     self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)
        
    #     self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
    #     self.eoi_toks = self._get_eoi_toks()
    #     self.refusal_toks = self._get_refusal_toks()

    #     self.model_block_modules = self._get_model_block_modules()
    #     self.model_attn_modules = self._get_attn_modules()
    #     self.model_mlp_modules = self._get_mlp_modules()

    def get_instruction_with_sys_prompt(self, instruction: str):
        return "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:".format(instruction)

    def _load_model(self, model_path, dtype=torch.float16):
        hf_token = Path("/work/jcaspary/.hf_token").read_text().strip()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model_str: str = "MiniCPM-V-2_6"
        model_path = f"openbmb/{model_str}"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval().cuda()
        model.to(device, dtype=torch.bfloat16)
        model.requires_grad_(False)  # Freeze model parameters
        if model.processor is None:
            model.processor = AutoProcessor.from_pretrained(model.config._name_or_path, trust_remote_code=True)
        return model

    def _load_tokenizer(self, model_path):
        model_str: str = "MiniCPM-V-2_6"
        model_path = f"openbmb/{model_str}"
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # type: ignore

        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")  # Ensure correct padding token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_minicpm, tokenizer=self.tokenizer, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(
        "Answer:", 
        add_special_tokens=False
    )

    def _get_refusal_toks(self):
        return CPM_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.llm.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.self_attn for block in self._get_model_block_modules()])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self._get_model_block_modules()])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_minicpmv_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_minicpmv_weights, direction=direction, coeff=coeff, layer=layer)
    
    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=1, max_new_tokens=64, is_vlm=False):
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        
        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]
        pixel_values = [d["pixel_values"] for d in dataset]
        #eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        #generation_config['eos_token_id'] = eos_token_id
        for i in tqdm(range(0, len(dataset), batch_size)):
            batched_pixel_values = pixel_values[i:i+batch_size]
            batched_instructions = instructions[i:i + batch_size]
            inputs,_ = tokenize_instructions_minicpm(self.tokenizer, batched_instructions, None, True, batched_pixel_values, self.model)
            inputs.to(self.model.device)
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                responses = []
                for j in range(0, len(batched_pixel_values)):
                    
                    response = self.model.generate(
                        **inputs,
                        tokenizer=self.tokenizer,
                        vision_hidden_states=None,
                        stream=False,
                        decode_text=True,
                        **generation_config
                    )
                    print(response)
                    responses.append(response[0])
                for j in range(0, len(batched_pixel_values)):
                    completions.append({
                        'category': categories[i + j],
                        'prompt': instructions[i + j],
                        'response': responses[j]
                    })
        return completions
