import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModel, GenerationConfig
from typing import List
from jaxtyping import Int, Float
from pipeline.utils.hook_utils import add_hooks
from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import torchvision.transforms as T
# Vicuna prompt format (Alpaca-style)
INTERNVL_REFUSAL_TOKS = [295]  # Example: 'I' (check if needed)

def orthogonalize_internvl_weights(basemodel, direction: Float[Tensor, "d_model"]):
    lm = basemodel.model.language_model.model

    # Embedding
    lm.tok_embeddings.weight.data = get_orthogonalized_matrix(
        lm.tok_embeddings.weight.data, direction
    )

    # Decoder layers: attention output + MLP down projection
    for block in lm.layers:
        block.attention.wo.weight.data = get_orthogonalized_matrix(
            block.attention.wo.weight.data.T, direction
        ).T

        block.feed_forward.w2.weight.data = get_orthogonalized_matrix(
            block.feed_forward.w2.weight.data.T, direction
        ).T

    # Optional: vision-language projector
    if hasattr(basemodel.model, "mlp1") and isinstance(basemodel.model.mlp1[-1], torch.nn.Linear):
        basemodel.model.mlp1[-1].weight.data = get_orthogonalized_matrix(
            basemodel.model.mlp1[-1].weight.data.T, direction
        ).T


def act_add_internvl_weights(basemodel, direction: Float[Tensor, "d_model"], coeff, layer):
    lm_layer = basemodel.model.language_model.model.layers[layer - 1]

    dtype = lm_layer.feed_forward.w2.weight.dtype
    device = lm_layer.feed_forward.w2.weight.device
    bias = (coeff * direction).to(dtype=dtype, device=device)

    # Modify MLP down projection bias
    lm_layer.feed_forward.w2.bias = torch.nn.Parameter(bias)

    # Optional: modify projector bias if exists
    if hasattr(basemodel.model, "mlp1") and isinstance(basemodel.model.mlp1[-1], torch.nn.Linear):
        basemodel.model.mlp1[-1].bias = torch.nn.Parameter(bias)

def format_instruction_internvl(
        instruction: str,
        output: str = None,
        include_trailing_whitespace: bool = True
    ):
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n')
        messages = []
        messages.append([roles[0], instruction])
        messages.append([roles[1], None])
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。"
        query = get_prompt_for_internvl2(system_message=system_message, messages=messages)
        formatted_instruction = query



        if not include_trailing_whitespace:
            formatted_instruction = formatted_instruction.rstrip()
        
        if output is not None:
            formatted_instruction += output

        return formatted_instruction

def load_image_from_image(image_file):
        transform = build_transform(input_size=448)
        images = dynamic_preprocess(image_file, image_size=448, use_thumbnail=True, max_num=12)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values


# TODO: Rewrite completly so its useable at every get_tokenize_instructions call
def tokenize_instructions_and_format_pixels_intern(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace=True,
    pixel_values=None,
    model=None,
):
    
    pixel_values_formatted = [load_image_from_image(d) for d in pixel_values]
    for i in range(len(instructions)):
        prompt = f"<image>\n{instructions[i]}"
        query = format_instruction_internvl(prompt)
        num_patches_list = [pixel_values_formatted[i].shape[0]]
        num_image_token = 256  # This is crucial!
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id
        eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * num_image_token * num_patches) + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        instructions[i] = query
    model_inputs = tokenizer(instructions, padding=True, truncation=False, return_tensors='pt')
    return model_inputs, pixel_values_formatted


def get_prompt_for_internvl2(system_message: str, messages: list) -> str:
                """
                Mimics `get_prompt()` for the InternVL2 template using SeparatorStyle.MPT.

                Args:
                    system_message (str): The system prompt (usually in Chinese for InternVL2).
                    messages (list): List of (role, message) tuples like:
                                    [('<|im_start|>user\n', 'Hi there'), ('<|im_start|>assistant\n', None)]

                Returns:
                    str: Full prompt string to feed into tokenizer/model.
                """
                sep = '<|im_end|>'
                ret = f"<|im_start|>system\n{system_message}{sep}"

                for role, message in messages:
                    if message:
                        # If message is a tuple, unpack it
                        if isinstance(message, tuple):
                            message = message[0]
                        ret += f"{role}{message}{sep}"
                    else:
                        ret += role  # usually ends with assistant turn

                return ret


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values




class InternVLModel(ModelBase):
    def __init__(self, model_path):
        self.gen_config = {}
        super().__init__(model_path)
        

    def _load_model(self, model_path, dtype=torch.float16):
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()

        model.requires_grad_(False)  # Freeze model parameters

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        tokenizer.padding_side = "left"
        #TODO: maybe use '<|im_end|>'
        tokenizer.pad_token = tokenizer.eos_token
        eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.gen_config['eos_token_id'] = eos_token_id
        self.gen_config["max_new_tokens"]=1
        self.gen_config["do_sample"]=False
        self.gen_config["return_dict_in_generate"]=True
        self.gen_config["output_scores"]=True
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_and_format_pixels_intern, tokenizer=self.tokenizer, include_trailing_whitespace=True, model=self.model)

    def _get_eoi_toks(self):
        return self.tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)

    def _get_refusal_toks(self):
        return INTERNVL_REFUSAL_TOKS

        # Returns the list of decoder blocks from the language model
    def _get_model_block_modules(self):
        return self.model.language_model.model.layers

    # Returns the list of attention modules (wo, wqkv, etc.)
    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.attention for block in self._get_model_block_modules()])

    # Returns the list of MLP modules (w1, w2, w3, act_fn)
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.feed_forward for block in self._get_model_block_modules()])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_internvl_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_internvl_weights, direction=direction, coeff=coeff, layer=layer)

    def get_instruction_with_sys_prompt(self, instruction: str):
        return ""

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64, is_vlm=False):
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]

        pixel_values = [self.load_image_from_image(d["pixel_values"]) for d in dataset]
        eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        generation_config['eos_token_id'] = eos_token_id
        for i in tqdm(range(0, len(dataset), batch_size)):
            batched_pixel_values = pixel_values[i:i+batch_size]
            batched_instructions = instructions[i:i + batch_size]
            inputs, pixel_values_formatted = tokenize_instructions_and_format_pixels_intern(self.tokenizer, batched_instructions, None, True, batched_pixel_values, self.model)
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                responses = []
                for j in range(0, len(batched_pixel_values)):
                    
                    response_toks = self.model.generate(
                        pixel_values=pixel_values_formatted[j].to(torch.bfloat16).to(self.model.device),
                        input_ids=inputs[j].input_ids.to(self.model.device),
                        attention_mask=inputs[j].attention_mask.to(self.model.device),
                        **generation_config
                    )
                    response = self.tokenizer.batch_decode(response_toks, skip_special_tokens=True)[0]
                    response = response.split('<|im_end|>')[0].strip()
                    responses.append(response)
                    # pdb.set_trace()
                for j in range(0, len(batched_pixel_values)):
                    completions.append({
                        'category': categories[i + j],
                        'prompt': instructions[i + j],
                        'response': responses[j]
                    })
                    #pdb.set_trace()
        return completions
