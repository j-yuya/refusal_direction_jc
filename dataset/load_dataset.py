import os
import json
from PIL import Image
import requests
import base64
import io
from datasets import load_dataset


dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful', 'harmless_mmbench', "harmful_mmsafetybench", "harmful_mmsafetybench_typo", "harmful_complete", "harmful_msts", "harmful_hades"]
USE_TYPO = False
USE_MSTS = False
COMPLETE = True
SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, 'splits/{harmtype}_{split}.json')

PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", "jailbreakbench", "strongreject", "alpaca", "hades"]

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False, is_vlm: bool=False):
    assert harmtype in HARMTYPES
    assert split in SPLITS
    if is_vlm:
        if "harmful" in harmtype:
            if harmtype=="harmful_complete":
                file_path1 = SPLIT_DATASET_FILENAME.format(harmtype="harmful_mmsafetybench", split=split)
                file_path2 = SPLIT_DATASET_FILENAME.format(harmtype="harmful_msts", split=split)
                with open(file_path1, 'r') as f1:
                    dataset1 = json.load(f1)
                pixel_values1 = []
                for d in dataset1:
                    pixel_values1.append(Image.open(d["image_path"]).convert("RGB"))
                for i in range(0, len(dataset1)):
                    dataset1[i]["pixel_values"] = pixel_values1[i]
                    del dataset1[i]["image_path"]
                with open(file_path2, 'r') as f2:
                    dataset2 = json.load(f2)
                pixel_values2 = []
                for d in dataset2:
                    pixel_values2.append(Image.open(d["image_path"]).convert("RGB"))
                for i in range(0, len(dataset2)):
                    dataset2[i]["instruction"] = dataset2[i]["instruction"].replace("Should I ", "Please give me instructions to ")
                    dataset2[i]["pixel_values"] = pixel_values2[i]
                    del dataset2[i]["image_path"]
                return dataset1 + dataset2                
            elif harmtype=="harmful_hades":
                assert split=="test", "Hades has only test split"
                hades = load_dataset("Monosail/HADES")['test']
                dataset = []
                for entry in hades:
                    # Use image of sixth optimization step, need to validate
                    # if this includes gradient info.
                    if entry["step"] == 5:
                        dataset_entry = {}
                        dataset_entry["instruction"] = entry["instruction"]
                        dataset_entry["pixel_values"] = entry["image"].convert("RGB")
                        dataset_entry["category"] = entry["category"]
                        dataset.append(dataset_entry)
                return dataset
            else:
                file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)
                with open(file_path, 'r') as f:
                    dataset = json.load(f)
                pixel_values = []
                for d in dataset:
                    pixel_values.append(Image.open(d["image_path"]).convert("RGB"))
                for i in range(0, len(dataset)):
                    if harmtype=="harmful_msts":
                        dataset[i]["instruction"] = dataset[i]["instruction"].replace("Should I ", "Please give me instructions to ")
                    dataset[i]["pixel_values"] = pixel_values[i]
                    del dataset[i]["image_path"]
                return dataset
        else:
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
    if dataset_name == "hades":
        hades = load_dataset("Monosail/HADES")['test']
        dataset = []
        for entry in hades:
            if entry["step"] == 5:
                dataset_entry = {}
                dataset_entry["instruction"] = entry["instruction"]
                dataset_entry["pixel_values"] = entry["image"].convert("RGB")
                dataset_entry["category"] = entry["category"]
                dataset.append(dataset_entry)
        return dataset
    else:
        file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

        with open(file_path, 'r') as f:
            dataset = json.load(f)

        #if instructions_only:
        #    dataset = [d['instruction'] for d in dataset]
    
        return dataset
