import os
import json
from PIL import Image
import requests
import base64
import io
dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful', 'harmless_mm', "harmful_mm"]
USE_TYPO = False

SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, 'splits/{harmtype}_{split}.json')

PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", "jailbreakbench", "strongreject", "alpaca"]

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False, is_vlm: bool=False):
    assert harmtype in HARMTYPES
    assert split in SPLITS
    if is_vlm:
        if harmtype == "harmful":
            if USE_TYPO:
                harmtype="mmsafetybench_typo"
            else:
                harmtype="mmsafetybench"
            file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)
            with open(file_path, 'r') as f:
                dataset = json.load(f)
            pixel_values = []
            for d in dataset:
                pixel_values.append(Image.open(d["image_path"]).convert("RGB"))
            for i in range(0, len(dataset)):
                dataset[i]["pixel_values"] = pixel_values[i]
                del dataset[i]["image_path"]
            return dataset
        else:
            harmtype="mmbench"
            file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)
            with open(file_path, 'r') as f:
                dataset = json.load(f)
            pixel_values = []
            for d in dataset:
                image_data = base64.b64decode(d["image_base64"])
                image = Image.open(io.BytesIO(image_data))
                if image.mode in ('RGBA', 'P'):
                    image = image.convert('RGB')
                pixel_values.append(image)
            for i in range(0, len(dataset)):
                dataset[i]["pixel_values"] = pixel_values[i]
                del dataset[i]["image_base64"]
            return dataset
    else:
        file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)
        # TODO: REWRITE
        with open(file_path, 'r') as f:
            dataset = json.load(f)
        #if instructions_only:
        #    dataset = [d['instruction'] for d in dataset]

        return dataset

def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    #if instructions_only:
    #    dataset = [d['instruction'] for d in dataset]
 
    return dataset
