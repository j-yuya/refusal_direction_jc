from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float
import torch
from pipeline.utils.hook_utils import add_hooks

class ModelBase(ABC):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)
        
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass

    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        pass

    @abstractmethod
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        pass

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64, is_vlm=False):
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        import pdb
        
        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]
        if is_vlm:
            # pixel_dtype = next(self.model.parameters()).dtype
            # image_transform = self.model.vision_backbone.image_transform
            # pixel_values = [image_transform(d["pixel_values"]).to(dtype=pixel_dtype) for d in dataset]
            pixel_values = [d["pixel_values"] for d in dataset]
            # pixel_values = torch.stack(pixel_values)

        for i in tqdm(range(0, len(dataset), batch_size)):
            
            if is_vlm:
                batched_pixel_values = pixel_values[i:i+batch_size]
            else: 
                tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i + batch_size])
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                if is_vlm:
                    #TODO: Add content of generate function of prismatic model to mimic behaviour.
                    batched_instructions = instructions[i:i + batch_size]
                    responses = []
                    import pdb
                    #pdb.set_trace()
                    for j in range(0, len(batched_pixel_values)):
                        # response = self.model.generate_batch(
                        #     pixel_values=batched_pixel_values[j].unsqueeze(0),
                        #     texts=[batched_instructions[j]],
                        #     generation_config=generation_config
                        # )
                        response= self.model.generate(image=batched_pixel_values[j], prompt_text=batched_instructions[j], generation_config=generation_config, min_length=1)
                        responses.append(response)
                        # pdb.set_trace()
                    for j in range(0, len(batched_pixel_values)):
                        completions.append({
                            'category': categories[i + j],
                            'prompt': instructions[i + j],
                            'response': responses[j]
                        })
                        #pdb.set_trace()
                else:
                    generation_toks = self.model.generate(
                        input_ids=tokenized_instructions.input_ids.to(self.model.device),
                        attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                        generation_config=generation_config,
                    )

                    generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

                    for generation_idx, generation in enumerate(generation_toks):
                        completions.append({
                            'category': categories[i + generation_idx],
                            'prompt': instructions[i + generation_idx],
                            'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                        })

        return completions
