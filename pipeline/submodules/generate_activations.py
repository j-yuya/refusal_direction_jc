import torch
import os
import requests
from typing import List, Dict
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from PIL import Image

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase

def get_collect_activations_pre_hook(layer, cache: Dict[int, List[torch.Tensor]], positions: List[int]):
    def hook_fn(module, input):
        # input[0] shape: (batch_size, seq_len, d_model)
        activation = input[0].detach().cpu()  # Don't need grads
        selected = activation[:, positions, :]  # shape: (batch, len(positions), d_model)

        # Store each sample's activations separately
        for i in range(selected.size(0)):
            cache[layer].append(selected[i])  # shape: (len(positions), d_model)

    return hook_fn


def get_all_activations(model, tokenizer, dataset, tokenize_instructions_fn, is_vlm, block_modules: List[torch.nn.Module], batch_size=1, positions=[-1], model_base = None):
    torch.cuda.empty_cache()

    if type(model).__name__ !='InternVLChatModel':
        n_layers = model.config.num_hidden_layers
    else: 
        n_layers = model.language_model.config.num_hidden_layers
    instructions = [d["instruction"] for d in dataset]

    if is_vlm and type(model).__name__ !='InternVLChatModel' and type(model).__name__ !='CogVLMForCausalLM' and type(model).__name__ !='MiniCPMV':
        pixel_dtype = next(model.parameters()).dtype
        image_transform = model.vision_backbone.image_transform
        pixel_values = [image_transform(d["pixel_values"]).to(dtype=pixel_dtype) for d in dataset]
        pixel_values = torch.stack(pixel_values)
    elif type(model).__name__ =='InternVLChatModel' or type(model).__name__ =='CogVLMForCausalLM' or type(model).__name__ =='MiniCPMV':
        pixel_values =  [(d["pixel_values"]) for d in dataset]
    # Initialize cache: {layer_idx: [activation_tensor_per_sample]}
    activation_cache: Dict[int, List[torch.Tensor]] = {layer: [] for layer in range(n_layers)}

    # Register hooks for all layers
    fwd_pre_hooks = [
        (block_modules[layer], get_collect_activations_pre_hook(layer=layer, cache=activation_cache, positions=positions))
        for layer in range(n_layers)
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        batched_pixel_values = []
        if is_vlm:
            batched_pixel_values = pixel_values[i:i+batch_size]
        
        if type(model).__name__ !='InternVLChatModel' and type(model).__name__ !='MiniCPMV':
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
                    image_flags=image_flags,
                )
            else:
                model(
                    input_ids=inputs.input_ids.to(model.device),
                )

    return activation_cache