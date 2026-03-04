"""Quick test: just tokenizer loading."""
import sys
print("Starting...", flush=True)
from transformers import AutoTokenizer
print("Importing AutoTokenizer done", flush=True)
t = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-360M-Instruct')
print(f"Done! Vocab size: {t.vocab_size}", flush=True)
