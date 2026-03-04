"""Quick end-to-end test of the VLA pipeline components."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Step 1: Importing modules...", flush=True)
from models.config import VLMConfig
from data.processors import get_tokenizer, get_image_processor
from vla.dataset import VLADataset, VLACollator
from vla.env_utils import make_env, get_obs_image
from vla.expert import get_expert_actions
from torch.utils.data import DataLoader
print("All imports OK", flush=True)

print("\n=== Step 2: Test environment ===", flush=True)
env = make_env(size=8)
obs, info = env.reset(seed=42)
img = get_obs_image(obs)
print(f"Obs image size: {img.size}, mode: {img.mode}", flush=True)
actions = get_expert_actions(env)
print(f"Expert actions: {actions[:5]}... (total: {len(actions)})", flush=True)
env.close()

print("\n=== Step 3: Test tokenizer ===", flush=True)
cfg = VLMConfig()
tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
print(f"Tokenizer vocab: {tokenizer.vocab_size}", flush=True)
print(f"Image token id: {tokenizer.image_token_id}", flush=True)

print("\n=== Step 4: Test image processor ===", flush=True)
image_processor = get_image_processor(cfg.vit_img_size, cfg.vit_img_size, False)
processed, ratio = image_processor(img)
print(f"Processed image shape: {processed.shape}, ratio: {ratio}", flush=True)

print("\n=== Step 5: Test dataset ===", flush=True)
ds = VLADataset('vla/trajectories_test', tokenizer, image_processor, cfg.mp_image_token_length)
print(f"Dataset size: {len(ds)}, actions: {ds.action_counts}", flush=True)

item = ds[0]
print(f"input_ids: {item['input_ids'].shape}", flush=True)
print(f"labels: {item['labels'].shape}", flush=True)
print(f"images: {item['images'].shape}", flush=True)

valid_labels = item['labels'][item['labels'] != -100]
print(f"Target action: '{tokenizer.decode(valid_labels)}'", flush=True)

print("\n=== Step 6: Test collator ===", flush=True)
collator = VLACollator(tokenizer, max_length=512)
loader = DataLoader(ds, batch_size=4, collate_fn=collator)
batch = next(iter(loader))
print(f"Batch input_ids: {batch['input_ids'].shape}", flush=True)
print(f"Batch labels: {batch['labels'].shape}", flush=True)
print(f"Batch images: {len(batch['images'])} items", flush=True)

print("\n=== ALL TESTS PASSED ===", flush=True)
