import torch
import random
import json
import os
import argparse
import platonic
from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores, get_representations
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.submodules.generate_activations import get_all_activations

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--cfg_template', type=str, required=False, default=None)
    parser.add_argument('--disable_images', action='store_false', dest='use_images')
    parser.add_argument('--image_type', type=str, required=False, default=None)
    return parser.parse_args()

def load_and_sample_datasets(cfg, is_vlm, image_type):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """

    harmless_train = load_dataset_split(harmtype=cfg.train_dataset_harmless, split='train', instructions_only=True, is_vlm=is_vlm, image_type=image_type, full_data=True)

    return harmless_train

def get_representations_outer(cfg, model_base, harmless_train, is_vlm, use_images):
    representations = get_representations(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, is_vlm=is_vlm, model_base=model_base, use_images=use_images)

    return representations

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

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None, is_vlm=False, use_images=True):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))
    # TODO: Use another evaluation set for Multimodal!
    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens, is_vlm=is_vlm, use_images=use_images)
    for completion in completions:
        print(completion["response"])
    # with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
    #     json.dump(completions, f, indent=4)

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

def run_pipeline(model_path, cfg_template, use_images, image_type):
    print("USE IMAGES:")
    print(use_images)
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    if cfg_template is not None:
        cfg.load_template(cfg_template)
    else:
        print("Using default cfg template")

    model_base = construct_model_base(cfg.model_path)
    is_vlm = cfg.is_vlm
    # Load and sample datasets
    harmless_train = load_and_sample_datasets(cfg, is_vlm, image_type)

    representations = get_representations_outer(cfg, model_base, harmless_train, is_vlm, use_images)
    
    if not os.path.exists(cfg.artifact_path()):
        os.makedirs(cfg.artifact_path())
    if use_images:
        if image_type is None:
            image_type = ""
        torch.save(representations, f'{cfg.artifact_path()}/representations_w_images_{image_type}.pt')
    else:
        torch.save(representations, f'{cfg.artifact_path()}/representations_text_only.pt')
    
    # baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    # generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', "harmless_train", harmless_train, is_vlm=is_vlm, use_images=use_images)

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, cfg_template=args.cfg_template, use_images=args.use_images, image_type=args.image_type)

