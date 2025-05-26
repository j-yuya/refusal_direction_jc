import torch
import random
import json
import os
import argparse

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.submodules.generate_activations import get_all_activations

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--cfg_template', type=str, required=False, default=None)
    parser.add_argument('--direction_dir', type=str, required=False, default=None)
    return parser.parse_args()

def load_and_sample_datasets(cfg, is_vlm):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    harmful_train =load_dataset_split(harmtype=cfg.train_dataset_harmful, split='train', instructions_only=True, is_vlm=is_vlm)
    return harmful_train

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val, is_vlm):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, is_vlm, model_base=model_base)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, is_vlm, model_base=model_base)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, is_vlm, model_base=model_base)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, is_vlm, model_base=model_base)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    return harmful_train, harmless_train, harmful_val, harmless_val

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train, is_vlm):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"),
        is_vlm=is_vlm)

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, is_vlm):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction"),
        is_vlm=is_vlm,
        kl_threshold=cfg.kl_threshold,
        induce_refusal_threshold=cfg.refusal_threshold
    )

    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')

    return pos, layer, direction

def load_direction(cfg):
    """Load direction metadata and tensor from the specified directory in cfg."""
    direction_path = cfg.direction_dir
    metadata_file = os.path.join(direction_path, 'direction_metadata.json')
    direction_file = os.path.join(direction_path, 'direction.pt')

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    pos = metadata["pos"]
    layer = metadata["layer"]

    # Load direction tensor
    direction = torch.load(direction_file)

    return pos, layer, direction

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None, is_vlm=False):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))
    # TODO: Use another evaluation set for Multimodal!
    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens, is_vlm=is_vlm)
    
    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def run_pipeline(model_path, cfg_template, direction_dir):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, direction_dir=direction_dir)
    if cfg_template is not None:
        cfg.load_template(cfg_template)
    else:
        print("Using default cfg template")

    model_base = construct_model_base(cfg.model_path)
    is_vlm = cfg.is_vlm
    # Load and sample datasets
    harmful_train = load_and_sample_datasets(cfg, is_vlm)
 
    pos, layer, direction = load_direction(cfg)
    
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)

    # 3a. Generate and save completions on harmful evaluation datasets
    harmful_test = load_dataset_split(harmtype=cfg.train_dataset_harmful, split='test', is_vlm=is_vlm)

    for dataset_name in cfg.evaluation_datasets:
        # generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', "harmful_train", harmful_train, is_vlm=is_vlm)
        # generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', "harmful_test", harmful_test, is_vlm=is_vlm)
        # generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', "harmful_train", harmful_train, is_vlm=is_vlm)
        # generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', "harmful_test", harmful_test, is_vlm=is_vlm)
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', "harmful_train", eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', "harmful_train", eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', "harmful_test", eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', "harmful_test", eval_methodologies=cfg.jailbreak_eval_methodologies)

    

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, cfg_template=args.cfg_template, direction_dir=args.direction_dir)

