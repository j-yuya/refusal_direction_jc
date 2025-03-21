from transformers import AutoTokenizer

# Load the Vicuna 7B tokenizer
tokenizer = AutoTokenizer.from_pretrained("/ceph/jcaspary/hf_cache/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d")

# Tokenize the word "I"
tokenized = tokenizer("I", add_special_tokens=False)

# Print token ID and token representation
print(f"Token ID: {tokenized.input_ids}")
print(f"Token: {tokenizer.convert_ids_to_tokens(tokenized.input_ids)}")