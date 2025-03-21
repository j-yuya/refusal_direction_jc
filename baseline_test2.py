from dataset.load_dataset import load_dataset_split
import random
from pipeline.model_utils.model_factory import construct_model_base
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from prismatic import load
import torch
from pathlib import Path
import json

hf_token = Path("/work/jcaspary/.hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


test_prismatic = False

harmful_test = random.sample(load_dataset_split(harmtype='harmful', split='test', is_vlm=True), 100)
harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test', is_vlm=True), 100)
datasets = [harmful_test, harmless_test]
batch_size = 32

completions = []

if not test_prismatic:
    model_base = construct_model_base("/ceph/jcaspary/hf_cache/hub/models--TRI-ML--prismatic-vlms/snapshots/a3ba8a19c453a82eaf5a3fb1e699dd9e441f0a12/reproduction-llava-v15+13b")

    generation_config = GenerationConfig(max_new_tokens=64, do_sample=False)
    generation_config.pad_token_id = model_base.tokenizer.pad_token_id
else:
    model_id = "reproduction-llava-v15+13b"
    #model_id = "mistral-instruct-v0.1+7b"
    vlm = load(model_id, hf_token=hf_token)
    vlm.to(device, dtype=torch.bfloat16)


for dataset in datasets:
    instructions = [x['instruction'] for x in dataset]
    categories = [x['category'] for x in dataset]

    # pixel_dtype = next(self.model.parameters()).dtype
    # image_transform = self.model.vision_backbone.image_transform
    # pixel_values = [image_transform(d["pixel_values"]).to(dtype=pixel_dtype) for d in dataset]
    pixel_values = [d["pixel_values"] for d in dataset]
    # pixel_values = torch.stack(pixel_values)

    for i in tqdm(range(0, len(dataset), batch_size)):
        batched_pixel_values = pixel_values[i:i+batch_size]
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
            if test_prismatic:
                prompt_builder = vlm.get_prompt_builder()
                prompt_builder.add_turn(role="human", message=batched_instructions[j])
                prompt_text = prompt_builder.get_prompt()
                print(prompt_text)
                response = vlm.generate(image=batched_pixel_values[j], prompt_text=prompt_text, do_sample=False, max_new_tokens=64)
                #print(f"Prompt: {batched_instructions[j]}")
                #print(f"Pixel Values Shape: {batched_pixel_values[j].shape}")
            else:
                model = model_base.model
                pixel_dtype = next(model.parameters()).dtype
                image_transform = model.vision_backbone.image_transform

                # Transform Image
                pixel_values = image_transform(batched_pixel_values[j]).to(dtype=pixel_dtype).unsqueeze(0)

                # === First Forward Pass ===
                # Tokenization using `model_base.tokenize_instructions_fn`
                tokenized_instructions = model_base.tokenize_instructions_fn(instructions=[batched_instructions[j]])

                # Prepare Inputs for First Forward Pass
                inputs1 = {
                    "input_ids": tokenized_instructions.input_ids.to(model.device),
                    "attention_mask": tokenized_instructions.attention_mask.to(model.device),
                    "pixel_values": pixel_values.to(model.device),
                }

                # Forward Pass 1
                input_ids1= tokenized_instructions.input_ids
                attention_mask1 = tokenized_instructions.attention_mask
                response1 = model(**inputs1)
                token1 = torch.argmax(response1.logits[0,-1,:]).item()
                token_str1 = model.llm_backbone.tokenizer.decode(token1).strip()
                # === Second Forward Pass ===
                # Build Prompt
                
                prompt_text = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:".format(batched_instructions[j])

                # Tokenization using `model.llm_backbone.tokenizer`
                tokenized_instruction2 = model.llm_backbone.tokenizer(prompt_text, return_tensors="pt")

                # Prepare Inputs for Second Forward Pass
                inputs2 = {
                    "input_ids": tokenized_instruction2["input_ids"].to(model.device),
                    "attention_mask": tokenized_instruction2["attention_mask"].to(model.device),
                    "pixel_values": pixel_values.to(model.device),  # Ensure pixel values are included
                }
                input_ids2= tokenized_instruction2.input_ids
                attention_mask2 = tokenized_instruction2.attention_mask

                # Forward Pass 2
                response2 = model(**inputs2)
                token2 = torch.argmax(response2.logits[0,-1,:]).item()
                token_str2 = model.llm_backbone.tokenizer.decode(token2).strip()

                response= model_base.model.generate(image=batched_pixel_values[j], prompt_text=batched_instructions[j], generation_config=generation_config)
                pdb.set_trace()



            responses.append(response1)
        for j in range(0, len(batched_pixel_values)):
            completions.append({
                'category': categories[i + j],
                'prompt': instructions[i + j],
                'response': responses[j]
            })

with open("baseline_test_results.json", "w", encoding="utf-8") as json_file:
    json.dump(completions, json_file, indent=4, ensure_ascii=False)

print("✅ JSON file saved successfully!")