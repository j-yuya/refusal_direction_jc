from PIL import Image
import numpy as np
import random
from strong_reject.evaluate import evaluate_dataset


def shuffle_image_patches(pil_image, num_patches=4):
    """
    Shuffle the upper part (1024x1024) of an image in patch-wise fashion.
    Keeps the bottom 300 pixels (typography area) unchanged.
    
    Args:
        pil_image (PIL.Image): Input image of size 1024x1324 (RGB).
        num_patches (int): Number of patches per row/column (n x n total patches).

    Returns:
        PIL.Image: New image with upper part shuffled and bottom (300px) unchanged.
    """
    assert pil_image.size == (1024, 1324), "Expected image size of 1024x1324"
    assert pil_image.mode == 'RGB', "Image must be in RGB mode"
    
    # Split the image
    img = pil_image.copy()
    upper = img.crop((0, 0, 1024, 1024))
    bottom = img.crop((0, 1024, 1024, 1324))

    # Convert upper part to numpy array
    upper_np = np.array(upper)
    patch_size = 1024 // num_patches
    patches = []

    # Extract patches
    for i in range(num_patches):
        for j in range(num_patches):
            patch = upper_np[i * patch_size:(i + 1) * patch_size,
                             j * patch_size:(j + 1) * patch_size]
            patches.append(patch)

    # Shuffle patches
    random.shuffle(patches)

    # Reassemble
    shuffled = np.zeros_like(upper_np)
    idx = 0
    for i in range(num_patches):
        for j in range(num_patches):
            shuffled[i * patch_size:(i + 1) * patch_size,
                     j * patch_size:(j + 1) * patch_size] = patches[idx]
            idx += 1

    # Create new shuffled upper image and combine with bottom
    shuffled_upper_img = Image.fromarray(shuffled)
    final_img = Image.new('RGB', (1024, 1324))
    final_img.paste(shuffled_upper_img, (0, 0))
    final_img.paste(bottom, (0, 1024))
    
    return final_img


def shuffle_text_instruction(prompt: str) -> str:
    """
    Shuffle a text prompt at the word level as done in the SI-Attack paper.
    
    Args:
        prompt (str): Original harmful instruction string.

    Returns:
        str: Shuffled version of the prompt.
    """
    words = prompt.strip().split()
    random.shuffle(words)
    return ' '.join(words)

def rate_jailbreak(prompt: str) -> float:
    eval = evaluate_dataset([prompt], ["strongreject_fine_tuned"])["score"]
    return eval