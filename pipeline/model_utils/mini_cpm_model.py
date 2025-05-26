import torch
import functools
from torch import Tensor
from transformers import AutoTokenizer, AutoProcessor
from typing import List, Optional, Tuple, Literal, Dict, Any, Union
from jaxtyping import Int, Float
from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.utils import get_orthogonalized_matrix
from prismatic.models.vlms.prismatic import PrismaticVLM
from transformers.utils import TensorType
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
    if pixel_values is not None:
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
        model_inputs = model.processor.tokenizer(instructions_processed, return_tensors=None, truncation=None, max_length=None)
        # padded_input_ids, padding_lengths = model.processor.pad(
        #     model_inputs["input_ids"],
        #     padding_side="left"
        # )
        input_ids_tensor = torch.tensor(model_inputs["input_ids"])
        batch_size = len(instructions)
        # Now create the attention mask
        attention_mask = input_ids_tensor.ne(0)
        model_inputs["attention_mask"] = attention_mask
        model_inputs = MiniCPMVBatchFeature(data={"input_ids": input_ids_tensor,
                                          "attention_mask": attention_mask,
                                          "tgt_sizes":[[] for _ in range(batch_size)],
                                          "pixel_values": [[] for _ in range(batch_size)],
                                          "image_bound": [[] for _ in range(batch_size)]})
        return model_inputs, None
    
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
    
    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=1, max_new_tokens=64, is_vlm=False, use_images = True):
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
            if use_images:
                inputs,_ = tokenize_instructions_minicpm(self.tokenizer, batched_instructions, None, True, batched_pixel_values, self.model)
            else:
                inputs,_ = tokenize_instructions_minicpm(self.tokenizer, batched_instructions, None, True, None, self.model)
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


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)

from transformers.utils import requires_backends, is_torch_dtype, is_torch_device
from transformers.image_processing_utils import BatchFeature
class MiniCPMVBatchFeature(BatchFeature):
    r"""
    Extend from BatchFeature for supporting various image size
    """
    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        if tensor_type is None:
            return self
        
        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

        def converter(value):
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    return tensor
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )


        for key, value in self.items():
            self[key] = recursive_converter(converter, value)
        return self
            
    def to(self, *args, **kwargs) -> "MiniCPMVBatchFeature":
        requires_backends(self, ["torch"])
        import torch

        def cast_tensor(v):
            # check if v is a floating point
            if torch.is_floating_point(v):
                # cast and send to device
                return v.to(*args, **kwargs)
            elif device is not None:
                return v.to(device=device)
            else:
                return v

        new_data = {}
        device = kwargs.get("device")
        # Check if the args are a device or a dtype
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # The first argument is a dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            new_data[k] = recursive_converter(cast_tensor, v)
        self.data = new_data
        return self
    

import torch
import torch.nn.functional as F
from typing import List, Optional, Union
import math

def preprocess_for_attack(
    images: List[List[torch.Tensor]],
    patch_size: int = 14,
    scale_resolution: int = 448,
    max_slice_nums: int = 9,
    slice_mode: bool = True,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
):
    def normalize(tensor, mean, std):
        mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
        return (tensor - mean) / std

    def ensure_divide(length, patch_size):
        return max(round(length / patch_size) * patch_size, patch_size)

    def find_best_resize(h, w):
        r = w / h
        new_h = int(scale_resolution / (r**0.5))
        new_w = int(new_h * r)
        return ensure_divide(new_h, patch_size), ensure_divide(new_w, patch_size)

    def get_sliced_grid(h, w):
        area = h * w
        
        ratio = h * w / (scale_resolution ** 2)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1 or not slice_mode:
            return None
        best_grid = (1, 1)
        log_ratio = (w / h).log() if isinstance(w, torch.Tensor) else math.log(w / h)
        min_error = float("inf")
        for i in [multiple - 1, multiple, multiple + 1]:
            if i <= 1 or i > max_slice_nums:
                continue
            for rows in range(1, i + 1):
                if i % rows == 0:
                    cols = i // rows
                    err = abs(log_ratio - torch.log(torch.tensor(cols / rows)))
                    if err < min_error:
                        best_grid = (cols, rows)
                        min_error = err
        return best_grid

    def split_tensor_to_patches(tensor, grid):
        C, H, W = tensor.shape
        cols, rows = grid
        patch_h = H // rows
        patch_w = W // cols
        patches = []
        for i in range(rows):
            for j in range(cols):
                patch = tensor[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                patches.append(patch)
        return patches

    def reshape_by_patch(image: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
        """
        Reshape a [C, H, W] image tensor into [C, patch_size, HW // patch_size]
        using unfold, replicating MiniCPMV logic faithfully.

        Args:
            image (torch.Tensor): Tensor of shape [3, H, W]
            patch_size (int): Patch size for unfolding (default: 14)

        Returns:
            torch.Tensor: Tensor of shape [3, patch_size, HW // patch_size]
        """
        assert image.ndim == 3 and image.shape[0] == 3, "Expected image shape [3, H, W]"
        unfolded = F.unfold(image.unsqueeze(0), kernel_size=patch_size, stride=patch_size)  # [1, C*P*P, N]
        C = image.shape[0]
        unfolded = unfolded.view(C, patch_size, patch_size, -1)  # [C, P, P, N]
        reshaped = unfolded.permute(0, 1, 3, 2).reshape(C, patch_size, -1)  # [C, P, P*N] -> [C, P, N*P]
        return reshaped

    all_pixel_values = []
    all_image_sizes = []
    all_tgt_sizes = []

    for img_list in images:
        pixel_values = []
        image_sizes = []
        tgt_sizes = []
        for img in img_list:
            C, H, W = img.shape
            image_sizes.append((W, H))
            grid = get_sliced_grid(H, W)

            if grid is None:
                new_h, new_w = find_best_resize(H, W)
                resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bicubic', align_corners=False).squeeze(0)
                patches = [resized]
            else:
                # resize to grid-compatible size
                new_h = ensure_divide(H, grid[1])
                new_w = ensure_divide(W, grid[0])
                resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bicubic', align_corners=False).squeeze(0)
                patches = split_tensor_to_patches(resized, grid)

            # ✅ Compute tgt_size once based on resized image BEFORE reshape
            H_patches = resized.shape[1] // patch_size
            W_patches = resized.shape[2] // patch_size
            tgt_size = torch.tensor((H_patches, W_patches), device=resized.device)

            patches = [normalize(p, mean, std) for p in patches]
            reshaped = [reshape_by_patch(p) for p in patches]
            tgt_sizes.extend([tgt_size] * len(reshaped))
            pixel_values.extend(reshaped)

        all_pixel_values.append(pixel_values)
        all_image_sizes.append(image_sizes)
        all_tgt_sizes.append(torch.stack(tgt_sizes) if tgt_sizes else torch.empty(0))

    return {
        "pixel_values": all_pixel_values,
        "image_sizes": all_image_sizes,
        "tgt_sizes": all_tgt_sizes,
    }

def remove_projection(x: torch.Tensor,          # (..., d_model)
                      direction: torch.Tensor   # (d_model,)
                      ) -> torch.Tensor:
    """
    Removes the component of `x` that lies along `direction`.
    Works for tensors with arbitrary leading dimensions.
    """
    # d = direction / (direction.norm() + 1e-8)
    d = d.to(dtype=x.dtype, device=x.device)

    # inner-product along the last dim → shape: (...)
    coeff = torch.einsum("...d,d->...", x, d)
    # keepdim so broadcast works when we subtract
    return x - coeff.unsqueeze(-1) * d
