"""Quick test script for VLA dataset pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.processors import get_tokenizer, get_image_processor
from models.config import VLMConfig
from vla.dataset import VLADataset, VLACollator
from torch.utils.data import DataLoader

cfg = VLMConfig()
tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
image_processor = get_image_processor(cfg.vit_img_size, cfg.vit_img_size, False)

dataset = VLADataset('vla/trajectories_test', tokenizer, image_processor,
                      cfg.mp_image_token_length, output_format='action_only', max_length=512)
print(f'Dataset size: {len(dataset)}')
print(f'Actions: {dataset.action_counts}')

item = dataset[0]
print(f'Input IDs shape: {item["input_ids"].shape}')
print(f'Labels shape: {item["labels"].shape}')
print(f'Images shape: {item["images"].shape}')
print(f'Attention mask shape: {item["attention_mask"].shape}')

# Decode to see what tokens look like
decoded_input = tokenizer.decode(item["input_ids"], skip_special_tokens=False)
print(f'Decoded input (first 200 chars): {decoded_input[:200]}...')

# Check labels
label_ids = item["labels"]
valid_labels = label_ids[label_ids != -100]
decoded_labels = tokenizer.decode(valid_labels, skip_special_tokens=True)
print(f'Target action: {decoded_labels}')

collator = VLACollator(tokenizer, max_length=512)
loader = DataLoader(dataset, batch_size=4, collate_fn=collator)
batch = next(iter(loader))
print(f'Batch input_ids: {batch["input_ids"].shape}')
print(f'Batch labels: {batch["labels"].shape}')
print(f'Batch images: {len(batch["images"])} items')
print('Dataset test PASSED!')
