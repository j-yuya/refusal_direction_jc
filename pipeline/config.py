
import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    model_alias: str
    model_path: str
    n_train_harmful: int = 100
    n_test_harmful: int = 100
    n_val_harmful: int = 32
    n_train_harmless: int = 100
    n_test_harmless: int = 100
    n_val_harmless: int = 32
    filter_train: bool = True
    filter_val: bool = True
    evaluation_datasets: Tuple[str] = ("jailbreakbench",)
    max_new_tokens: int = 512
    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching",)
    refusal_eval_methodologies: Tuple[str] = ("substring_matching",)
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048
    train_dataset_harmful: str = "harmful"
    train_dataset_harmless: str = "harmless"
    is_vlm: bool = False
    kl_threshold: float = 0.1
    refusal_threshold: float = 0

    def artifact_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias, f"{self.train_dataset_harmful}_{self.train_dataset_harmless}", f"{self.kl_threshold}_{self.refusal_threshold}", f"n_train_harmful_{self.n_train_harmful}")
    
    def load_template(self, template_name: str):
        if template_name=="vlm_complete":
            self.n_train_harmful=800
            self.train_dataset_harmful="harmful_complete"
            self.train_dataset_harmless="harmless_mmbench"
            self.is_vlm=True
            self.kl_threshold=0.1
            self.refusal_threshold=-4
        elif template_name=="vlm_complete2":
            self.n_train_harmful=100
            self.train_dataset_harmful="harmful_complete"
            self.train_dataset_harmless="harmless_mmbench"
            self.is_vlm=True
            self.kl_threshold=0.1
            self.refusal_threshold=0
        elif template_name=="hades_jailbreak":
            self.train_dataset_harmful="harmful_hades"
            self.evaluation_datasets = ("hades",)
            self.is_vlm=True
        elif template_name=="figstep_jailbreak":
            self.train_dataset_harmful="harmful_figstep"
            self.evaluation_datasets = ("figstep",)
            self.is_vlm=True
        elif template_name=="hades_jailbreak_shuffled":
            self.train_dataset_harmful="harmful_hades"
            self.evaluation_datasets = ("hades_shuffled",)
            self.is_vlm=True
        else:
            print("WARNING: Cfg-Template unknown, using default template")

