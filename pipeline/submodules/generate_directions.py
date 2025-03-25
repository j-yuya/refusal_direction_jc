import torch
import os
import requests
from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from PIL import Image
from transformers import GenerationConfig

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase
import pdb
def get_mean_activations_pre_hook(layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn

def get_mean_activations(model, tokenizer, dataset, tokenize_instructions_fn, is_vlm, block_modules: List[torch.nn.Module], batch_size=1, positions=[-1], model_base=None):
    torch.cuda.empty_cache()
    n_positions = len(positions)
    n_samples = len(dataset)
    if type(model).__name__ !='InternVLChatModel':
        n_layers = model.config.num_hidden_layers
        d_model = model.config.hidden_size
    else:
        n_layers = model.language_model.config.num_hidden_layers
        d_model = model.language_model.config.hidden_size

    instructions = [d["instruction"] for d in dataset]
    if is_vlm and type(model).__name__ !='InternVLChatModel':
        pixel_dtype = next(model.parameters()).dtype
        image_transform = model.vision_backbone.image_transform
        pixel_values = [image_transform(d["pixel_values"]).to(dtype=pixel_dtype) for d in dataset]
        pixel_values = torch.stack(pixel_values)
    elif type(model).__name__ =='InternVLChatModel':
        pixel_values =  [(d["pixel_values"]) for d in dataset]

    # we store the mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    fwd_pre_hooks = [(block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions)) for layer in range(n_layers)]

    for i in tqdm(range(0, len(instructions), batch_size)):
        batched_pixel_values = []
        if is_vlm:
            batched_pixel_values = pixel_values[i:i+batch_size]
        
        if type(model).__name__ !='InternVLChatModel':
            inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        else:
            inputs, batched_pixel_values = tokenize_instructions_fn(instructions=instructions[i:i+batch_size], pixel_values=batched_pixel_values)
            batched_pixel_values = [pixels.to(dtype=torch.bfloat16) for pixels in batched_pixel_values]
            batched_pixel_values = torch.stack(batched_pixel_values)
        
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            if is_vlm and type(model).__name__ !='InternVLChatModel':
                model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                pixel_values=batched_pixel_values.to(model.device)
            )
            elif is_vlm and type(model).__name__ =='InternVLChatModel':
                generation_config = model_base.gen_config
                image_flags = torch.tensor([1] * batched_pixel_values[0].shape[0], dtype=torch.long).to(model.device)
                model(
                    pixel_values=batched_pixel_values.squeeze(0).to(model.device),
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                    image_flags=image_flags
                )
            else:
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )

    return mean_activations

def get_mean_diff(model, tokenizer, harmful_instructions, harmless_instructions, tokenize_instructions_fn, is_vlm, block_modules: List[torch.nn.Module], batch_size=1, positions=[-1], model_base=None):
    mean_activations_harmful = get_mean_activations(model, tokenizer, harmful_instructions, tokenize_instructions_fn, is_vlm, block_modules, batch_size=batch_size, positions=positions, model_base=model_base)
    mean_activations_harmless = get_mean_activations(model, tokenizer, harmless_instructions, tokenize_instructions_fn, is_vlm, block_modules, batch_size=batch_size, positions=positions, model_base=model_base)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_harmful - mean_activations_harmless

    return mean_diff

def generate_directions(model_base: ModelBase, harmful_instructions, harmless_instructions, artifact_dir, is_vlm):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    mean_diffs = get_mean_diff(model_base.model, model_base.tokenizer, harmful_instructions, harmless_instructions, model_base.tokenize_instructions_fn, is_vlm, model_base.model_block_modules, positions=list(range(-len(model_base.eoi_toks), 0)), model_base=model_base)
    if type(model_base.model).__name__ !='InternVLChatModel':
        assert mean_diffs.shape == (len(model_base.eoi_toks), model_base.model.config.num_hidden_layers, model_base.model.config.hidden_size)
        assert not mean_diffs.isnan().any()
    else:
        assert mean_diffs.shape == (len(model_base.eoi_toks), model_base.model.language_model.config.num_hidden_layers, model_base.model.language_model.config.hidden_size)
        assert not mean_diffs.isnan().any()
    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")

    return mean_diffs