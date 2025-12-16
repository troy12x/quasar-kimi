import os
import sys
import torch
import warnings
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    set_seed,
    get_scheduler
)
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import bitsandbytes as bnb
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# Disable cache warnings
warnings.filterwarnings("ignore")

class WikiText1MDataset(IterableDataset):
    def __init__(self, tokenizer, seq_length=1_048_576, infinite=True):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.infinite = infinite
        
        print("Loading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(dataset["text"])
        self.tokens = tokenizer.encode(text)
        print(f"Loaded {len(self.tokens):,} tokens from WikiText-2")
        
        # If dataset is smaller than seq_len, repeat it
        if len(self.tokens) < seq_length:
            repeat_factor = (seq_length // len(self.tokens)) + 2
            self.tokens = self.tokens * repeat_factor
            print(f"Expanded to {len(self.tokens):,} tokens for chunks")

    def __iter__(self):
        idx = 0
        while True:
            if idx + self.seq_length + 1 > len(self.tokens):
                if self.infinite:
                    idx = 0
                else:
                    break
            
            chunk = self.tokens[idx : idx + self.seq_length]
            labels = self.tokens[idx + 1 : idx + self.seq_length + 1] # Shift right
            
            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.ones(self.seq_length, dtype=torch.long)
            }
            
            idx += self.seq_length

def main():
    # Setup
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision='bf16',
        kwargs_handlers=[ddp_kwargs]
    )
    set_seed(42)

    # Config matching your 2M training
    model_name = "silx-ai/Quasar-2M-Base"
    SEQ_LEN = 1_048_576 # 1M tokens
    
    if accelerator.is_main_process:
        print(f"DEBUG: Starting WikiText check with {SEQ_LEN:,} seq length")
        print(f"DEBUG: Memory before load: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB (GPU)")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load Model (CPU First)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.model_max_length = SEQ_LEN
    config.use_cache = False
    
    if accelerator.is_main_process:
        print("Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()

    # Optimizer (8-bit Adam)
    print("Initializing 8-bit Adam...")
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-5)

    # Dataloader
    dataset = WikiText1MDataset(tokenizer, seq_length=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=1)

    # Prepare
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    # Training Loop
    model.train()
    if accelerator.is_main_process:
        print("\nStarting Training Loop...")
    
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            if accelerator.is_main_process:
                print(f"Step {step}: Loss = {loss.item():.4f}")
        
        if step >= 5: # Debug: Stop after 5 steps
            break

    if accelerator.is_main_process:
        print("Success! Training loop works on WikiText.")

if __name__ == "__main__":
    main()
