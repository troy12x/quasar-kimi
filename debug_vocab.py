
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_name = "silx-ai/Quasar-2M-Base"
print(f"Loading {model_name}...")

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
print(f"Config vocab size: {config.vocab_size}")
print(f"Config pad_token_id: {config.pad_token_id}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
print(f"Tokenizer len: {len(tokenizer)}")

# We won't load the full model to save time if config is enough, but to be sure let's load it on cpu (meta device if possible or just normal)
# Since we have high mem, loading on CPU is fine.
# But wait, looking at the error, we need to know the actual weight shape.

try:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    print(f"Model embedding shape: {model.get_input_embeddings().weight.shape}")
    
    if config.pad_token_id is not None:
        if config.pad_token_id >= model.get_input_embeddings().weight.shape[0]:
            print("ISSUE DETECTED: pad_token_id >= embedding size")
        else:
            print("pad_token_id is within embedding size")
except Exception as e:
    print(f"Error loading model: {e}")
