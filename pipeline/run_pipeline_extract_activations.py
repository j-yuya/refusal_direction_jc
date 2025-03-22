import torch
import random
import json
import os
import argparse

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_activations import get_all_activations
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()

def load_and_sample_datasets(cfg, is_vlm):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True, is_vlm=is_vlm), 100)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True, is_vlm=is_vlm), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True, is_vlm=is_vlm), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True, is_vlm=is_vlm), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val, is_vlm):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, is_vlm)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, is_vlm)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, is_vlm)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, is_vlm)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    return harmful_train, harmless_train, harmful_val, harmless_val



def run_pipeline(model_path, is_vlm):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg, is_vlm)
    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val, is_vlm)
    # 1. Generate candidate refusal directions
    print("length of harmful set")
    print(len(harmful_train))
    if not os.path.exists(cfg.artifact_path()):
        os.makedirs(cfg.artifact_path())
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'all_activations')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'all_activations'))

    all_activations_harmful = get_all_activations(model=model_base.model, tokenizer=model_base.tokenizer, dataset=harmful_train, tokenize_instructions_fn=model_base.tokenize_instructions_fn, is_vlm=is_vlm, block_modules=model_base.model_block_modules, positions=list(range(-len(model_base.eoi_toks), 0)))
 
    torch.save(all_activations_harmful, os.path.join(cfg.artifact_path(), 'all_activations/all_activations_harmful.pt'))

    all_activations_harmless = get_all_activations(model=model_base.model, tokenizer=model_base.tokenizer, dataset=harmless_train, tokenize_instructions_fn=model_base.tokenize_instructions_fn, is_vlm=is_vlm, block_modules=model_base.model_block_modules, positions=list(range(-len(model_base.eoi_toks), 0)))
 
    torch.save(all_activations_harmless, os.path.join(cfg.artifact_path(), 'all_activations/all_activations_harmless.pt'))

if __name__ == "__main__":
    args = parse_arguments()
    if args.model_path is None:
        run_pipeline(model_path="/ceph/jcaspary/hf_cache/hub/models--TRI-ML--prismatic-vlms/snapshots/a3ba8a19c453a82eaf5a3fb1e699dd9e441f0a12/reproduction-llava-v15+7b", is_vlm=True)
    else:
        model_path = args.model_path
        if "vlm" in model_path:
            run_pipeline(model_path=args.model_path, is_vlm=True)
        else:
            run_pipeline(model_path=args.model_path, is_vlm=False)
