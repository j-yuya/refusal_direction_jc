import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_image_from_image(image_file):
    transform = build_transform(input_size=448)
    images = dynamic_preprocess(image_file, image_size=448, use_thumbnail=True, max_num=12)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = '/ceph/jcaspary/hf_cache/hub/models--OpenGVLab--InternVL2-8B/snapshots/6f6d72be3c7a8541d2942691c46fbd075c147352'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
#pixel_values = load_image('/work/jcaspary/AstraFellowship-When-Do-VLM-Image-Jailbreaks-Transfer/images/trina/000.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=10, do_sample=False)


# pure-text conversation (纯文本对话)
# question = 'Hello, who are you?'
# response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Can you tell me a story?'
# response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
#print(f'User: {question}\nAssistant: {response}')

# single-image single-round conversation (单图单轮对话)
# question = '<image>\nPlease describe the image shortly.'
# response = model.chat(tokenizer, pixel_values, question, generation_config)
# print(f'User: {question}\nAssistant: {response}')

# # single-image multi-round conversation (单图多轮对话)
# question = '<image>\nPlease describe the image in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Please write a poem according to the image.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# question = '<image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

# question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list,
#                                history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # batch inference, single image per sample (单图批处理)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
# responses = model.batch_chat(tokenizer, pixel_values,
#                              num_patches_list=num_patches_list,
#                              questions=questions,
#                              generation_config=generation_config)
# for question, response in zip(questions, responses):
#     print(f'User: {question}\nAssistant: {response}')

from dataset.load_dataset import load_dataset_split
import random
from pipeline.model_utils.model_factory import construct_model_base
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from pathlib import Path
import json

hf_token = Path("/work/jcaspary/.hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



harmful_test = random.sample(load_dataset_split(harmtype='harmful_complete', split='train', is_vlm=True), 100)
harmless_test = random.sample(load_dataset_split(harmtype='harmless_mmbench', split='train', is_vlm=True), 100)
datasets = [harmless_test, harmful_test]
batch_size = 32

completions = []

for dataset in datasets:
    instructions = [x['instruction'] for x in dataset]
    categories = [x['category'] for x in dataset]
    pixel_values = [load_image_from_image(d["pixel_values"]) for d in dataset]

    for i in tqdm(range(0, len(dataset), batch_size)):
        batched_pixel_values = pixel_values[i:i+batch_size]
        #TODO: Add content of generate function of prismatic model to mimic behaviour.
        batched_instructions = instructions[i:i + batch_size]
        responses = []
        import pdb
        #pdb.set_trace()
        for j in range(0, len(batched_pixel_values)):
            prompt = f"<image>\n{batched_instructions[j]}"
            generation_config = dict(max_new_tokens=1024, do_sample=False)
            response= model.chat(tokenizer, pixel_values[j].to(torch.bfloat16).cuda(), prompt, generation_config)
            responses.append(response)

            # Tokenization using `model.llm_backbone.tokenizer`
            
            system_message = model.system_message
            generation_config = dict(max_new_tokens=10, do_sample=False, return_dict_in_generate=True, output_scores=True)

            def get_prompt_for_internvl2(system_message: str, messages: list) -> str:
                """
                Mimics `get_prompt()` for the InternVL2 template using SeparatorStyle.MPT.

                Args:
                    system_message (str): The system prompt (usually in Chinese for InternVL2).
                    messages (list): List of (role, message) tuples like:
                                    [('<|im_start|>user\n', 'Hi there'), ('<|im_start|>assistant\n', None)]

                Returns:
                    str: Full prompt string to feed into tokenizer/model.
                """
                sep = '<|im_end|>'
                ret = f"<|im_start|>system\n{system_message}{sep}"

                for role, message in messages:
                    if message:
                        # If message is a tuple, unpack it
                        if isinstance(message, tuple):
                            message = message[0]
                        ret += f"{role}{message}{sep}"
                    else:
                        ret += role  # usually ends with assistant turn

                return ret
            roles=('<|im_start|>user\n', '<|im_start|>assistant\n')
            messages = []
            messages.append([roles[0], prompt])
            messages.append([roles[1], None])

            query = get_prompt_for_internvl2(system_message=system_message, messages=messages)
            # Assume: pixel_values[j] has shape [num_patches, C, H, W]
            num_patches_list = [pixel_values[j].shape[0]]
            num_image_token = 256  # This is crucial!

            # 1. Insert correct number of IMG_CONTEXT tokens
            IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
            img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            model.img_context_token_id = img_context_token_id
            eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
            IMG_START_TOKEN = '<img>'
            IMG_END_TOKEN = '</img>'
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * num_image_token * num_patches) + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)

            

            # 2. Tokenize
            model_inputs = tokenizer(query, return_tensors='pt')
            input_ids2 = model_inputs.input_ids.to(model.device)

            n_img_tokens = input_ids2.eq(tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')).sum().item()
            image_flags = torch.ones(n_img_tokens, dtype=torch.long)

            attention_mask2 = model_inputs.attention_mask.to(model.device)
            generation_config['eos_token_id'] = eos_token_id
            # 3. Generate image_flags mask
            img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            #image_flags = (input_ids2 == img_context_token_id).long().unsqueeze(-1).to(model.device)
            image_flags = torch.tensor([1] * num_patches, dtype=torch.long).to(model.device)
            # 4. Convert pixel_values to correct dtype/device
            pixel_values2 = pixel_values[j].to(torch.bfloat16).to(model.device)

            #pdb.set_trace()
            # 5. Forward
            response2 = model.generate(
                pixel_values=pixel_values2,
                input_ids=input_ids2,
                attention_mask=attention_mask2,
                **generation_config
            )

            output=model(pixel_values=pixel_values2, input_ids=input_ids2, attention_mask=attention_mask2, image_flags=image_flags)

            last_token_logits = response2.scores[0]
            predicted_token_id = torch.argmax(last_token_logits).item()
            predicted_token = tokenizer.decode([predicted_token_id])
            print(predicted_token) 
            response3 = tokenizer.batch_decode(response2.sequences, skip_special_tokens=True)[0]
            response3 = response3.split('<|im_end|>')[0].strip()
            pdb.set_trace()

            
            # token2 = torch.argmax(response2.logits[0,-1,:]).item()
            # token_str2 = model.llm_backbone.tokenizer.decode(token2).strip()
            # pdb.set_trace()






        for j in range(0, len(batched_pixel_values)):
            completions.append({
                'category': categories[i + j],
                'prompt': instructions[i + j],
                'response': responses[j]
            })

with open("baseline_internvl27b_test_results.json", "w", encoding="utf-8") as json_file:
    json.dump(completions, json_file, indent=4, ensure_ascii=False)

print("✅ JSON file saved successfully!")