import torch
import functools
from torch import Tensor
from transformers import AutoTokenizer
from typing import List
from jaxtyping import Int, Float
from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.utils import get_orthogonalized_matrix
from prismatic.models.vlms.prismatic import PrismaticVLM
#from prismatic.models.backbones.llm import HFCausalLLMBackbone
#from prismatic.models.backbones.vision import VisionBackbone
from prismatic import load
from pathlib import Path

# Vicuna prompt format (Alpaca-style)
LLAVA_CHAT_TEMPLATE = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {instruction} ASSISTANT:"
LLAVA_REFUSAL_TOKS  = [306]  # Example: 'I' (check if needed)

def format_instruction_llava(
    instruction: str,
    output: str = None,
    include_trailing_whitespace: bool = True
):
    formatted_instruction = LLAVA_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_llava(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_llava(instruction, output, include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_llava(instruction, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_llava_weights(basemodel, direction: Float[Tensor, "d_model"]):
    """
    Applies orthogonalization to LLaVA weights, including LLM, attention, MLP, and projector layers.
    """
    # Modify LLM embedding weights
    basemodel.model.llm_backbone.llm.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        basemodel.model.llm_backbone.llm.model.embed_tokens.weight.data, direction
    )

    # Modify Attention & MLP weights
    for block in basemodel.model.llm_backbone.llm.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

    # Modify Vision-Language Projector weights
    basemodel.model.projector.weight.data = get_orthogonalized_matrix(basemodel.model.projector.weight.data.T, direction).T

def act_add_llava_weights(basemodel, direction: Float[Tensor, "d_model"], coeff, layer):
    """
    Applies activation addition to LLaVA weights by modifying biases in both MLP and projector layers.
    """
    dtype = basemodel.model.llm_backbone.llm.model.layers[layer - 1].mlp.down_proj.weight.dtype
    device = basemodel.model.llm_backbone.llm.model.layers[layer - 1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    # Modify MLP down projection bias
    basemodel.model.llm_backbone.llm.model.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)

    # Modify Vision-Language Projector bias (optional)
    bademodel.model.projector.bias = torch.nn.Parameter(bias)

class LlavaModel(ModelBase):
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

    def _load_model(self, model_path, dtype=torch.float16):
        """
        Loads the LLaVA 1.5 model, including both the vision backbone and Vicuna LLM.
        """

        hf_token = Path("/work/jcaspary/.hf_token").read_text().strip()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load Vision Backbone (SigLIP, CLIP, or ViT)
        #vision_backbone = VisionBackbone("siglip-vit-l", "resize-naive")

        # Load LLM Backbone (Vicuna 7B)
        #llm_backbone = HFCausalLLMBackbone("vicuna-7b", "llama", model_path)

        # Load LLaVA Model with Prismatic's architecture
        model = load("reproduction-llava-v15+7b", hf_token=hf_token).eval()

        model.requires_grad_(False)  # Freeze model parameters
        import pdb
        pdb.set_trace()
        return model

    def _load_tokenizer(self, model_path):
        """
        Loads the tokenizer for LLaVA, ensuring proper tokenization for multimodal prompts.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            "/ceph/jcaspary/hf_cache/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d",
            trust_remote_code=True,
            use_fast=False
        )

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token  # Ensure correct padding token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_llava, tokenizer=self.tokenizer, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(
        LLAVA_CHAT_TEMPLATE.split("{instruction}")[-1], 
        add_special_tokens=False
    )

    def _get_refusal_toks(self):
        return LLAVA_REFUSAL_TOKS  # Same as Vicuna

    def _get_model_block_modules(self):
        return self.model.llm_backbone.llm.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.self_attn for block in self._get_model_block_modules()])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self._get_model_block_modules()])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_llava_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_llava_weights, direction=direction, coeff=coeff, layer=layer)
