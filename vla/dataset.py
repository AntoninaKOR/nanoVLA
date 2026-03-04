"""
VLA Dataset: loads expert trajectories for SFT and GRPO training.

Converts (observation_image, action) pairs into the nanoVLM format:
- Image processed through the VLM's image processor
- Text formatted as chat messages with action as the assistant response
"""
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from vla.env_utils import ACTION_NAMES
from data.processors import get_image_string


class VLADataset(Dataset):
    """
    Dataset of (image, action) pairs from expert trajectories.

    Supports two output formats:
      - "action_only": assistant outputs just the action token
      - "cot": assistant outputs a reasoning description + action
    """

    PROMPT = "You are a navigation agent in a grid world. Given the current observation, output the next action to reach the green goal square."

    def __init__(self, data_dir, tokenizer, image_processor, mp_image_token_length,
                 output_format="action_only", max_length=512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.output_format = output_format
        self.max_length = max_length

        # Load step records
        with open(self.data_dir / "steps.json") as f:
            self.steps = json.load(f)

        # Pre-compute action distribution for logging
        action_counts = {}
        for s in self.steps:
            a = s["action"]
            action_counts[a] = action_counts.get(a, 0) + 1
        self.action_counts = action_counts

    def __len__(self):
        return len(self.steps)

    def _format_assistant_response(self, action_name, step_info=None):
        """Format the assistant response based on output format."""
        if self.output_format == "action_only":
            return action_name
        else:  # cot format
            step_id = step_info.get("step_id", 0) if step_info else 0
            total = step_info.get("total_steps_in_episode", 1) if step_info else 1
            progress = f"Step {step_id + 1} of {total}."
            return f"{progress} Action: {action_name}"

    def __getitem__(self, idx):
        step = self.steps[idx]
        action_name = step["action"]

        # Load and process image
        img_path = self.data_dir / step["image_path"]
        img = Image.open(img_path).convert("RGB")
        processed_image, splitted_image_ratio = self.image_processor(img)

        # Build image string
        image_string = get_image_string(
            self.tokenizer, [splitted_image_ratio], self.mp_image_token_length
        )

        # Format messages
        user_content = image_string + self.PROMPT
        assistant_content = self._format_assistant_response(action_name, step)

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        # Tokenize
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Create labels: mask everything except the assistant response
        prompt_messages = [{"role": "user", "content": user_content}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # Labels: -100 for prompt tokens, actual ids for response tokens
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        # Truncate if needed
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "images": processed_image,
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        }


class VLACollator:
    """Collate VLA samples with left-padding."""

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        max_len = min(max(len(b["input_ids"]) for b in batch), self.max_length)

        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        images_list = []

        for b in batch:
            seq_len = len(b["input_ids"])
            pad_len = max_len - seq_len

            if pad_len > 0:
                # Left-pad
                input_ids = torch.cat([
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    b["input_ids"]
                ])
                labels = torch.cat([
                    torch.full((pad_len,), -100, dtype=torch.long),
                    b["labels"]
                ])
                attention_mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    b["attention_mask"]
                ])
            else:
                input_ids = b["input_ids"][:max_len]
                labels = b["labels"][:max_len]
                attention_mask = b["attention_mask"][:max_len]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            images_list.append(b["images"])

        return {
            "input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
            "attention_mask": torch.stack(attention_mask_list),
            "images": images_list,
        }
