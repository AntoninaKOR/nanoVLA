"""
SFT (Supervised Fine-Tuning) training for VLA.

Fine-tunes NanoVLM on expert trajectories from MiniGrid EmptyEnv
to predict actions from visual observations.
"""
import os
import sys
import math
import time
import json
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from comet_ml import Experiment
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
from data.processors import get_tokenizer, get_image_processor

from vla.dataset import VLADataset, VLACollator
from vla.evaluate import evaluate_in_env

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def get_lr(step, max_lr, max_steps):
    """Cosine learning rate schedule with linear warmup."""
    min_lr = max_lr * 0.1
    warmup_steps = int(max_steps * 0.03)
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_sft(args):
    # Device
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Load model
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = VisionLanguageModel.from_pretrained(args.checkpoint)
    else:
        print(f"Loading model from HuggingFace: {args.hf_model}")
        model = VisionLanguageModel.from_pretrained(args.hf_model)

    cfg = model.cfg
    model.to(device)
    model.train()

    # Tokenizer and image processor
    tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    # Use vit_img_size as max to keep MiniGrid images as single patch
    image_processor = get_image_processor(cfg.vit_img_size, cfg.vit_img_size, False)

    # Dataset
    dataset = VLADataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=cfg.mp_image_token_length,
        output_format=args.output_format,
        max_length=args.max_length,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Action distribution: {dataset.action_counts}")

    # Train/val split
    val_size = min(int(len(dataset) * 0.1), 2000)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    collator = VLACollator(tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=1, pin_memory=True,
    )

    # Optimizer - different LR for different modules
    param_groups = [
        {"params": list(model.MP.parameters()), "lr": args.lr_mp},
        {"params": list(model.vision_encoder.parameters()), "lr": args.lr_backbone},
        {"params": list(model.decoder.parameters()), "lr": args.lr_backbone},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)

    # Comet ML
    experiment = Experiment(project_name=args.comet_project, auto_metric_logging=False)
    experiment.set_name(f"sft_{args.output_format}")
    experiment.log_parameters(vars(args))

    # Training loop
    max_steps = args.max_steps
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    eval_results = []

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nStarting SFT training for {max_steps} steps")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum_steps}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"  Output format: {args.output_format}")

    epoch = 0
    pbar = tqdm(total=max_steps, desc="SFT Training", unit="step")
    while global_step < max_steps:
        epoch += 1
        model.train()
        epoch_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            # Update learning rate
            for pg in optimizer.param_groups:
                base_lr = pg["lr"]
                # Scale relative to step
                lr_scale = get_lr(global_step, 1.0, max_steps)
                # actual lr = base_lr from param_group init * scale  
                # But we set lr directly
            lr_mp = get_lr(global_step, args.lr_mp, max_steps)
            lr_bb = get_lr(global_step, args.lr_backbone, max_steps)
            optimizer.param_groups[0]["lr"] = lr_mp
            optimizer.param_groups[1]["lr"] = lr_bb
            optimizer.param_groups[2]["lr"] = lr_bb

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"]

            autocast_dtype = torch.bfloat16 if device.type in ["cuda", "cpu"] else torch.float16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
                loss = loss / args.grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = loss.item() * args.grad_accum_steps
                epoch_loss += step_loss
                num_batches += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{lr_bb:.2e}")

                if global_step % args.log_interval == 0:
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                    tqdm.write(f"  Step {global_step}/{max_steps} | Loss: {step_loss:.4f} | Avg: {avg_loss:.4f} | LR: {lr_bb:.2e}")
                    train_losses.append({"step": global_step, "loss": step_loss})
                    experiment.log_metrics({"train_loss": step_loss, "avg_loss": avg_loss, "lr": lr_bb}, step=global_step)

                # Validation
                if global_step % args.eval_interval == 0:
                    val_loss = validate(model, val_loader, device)
                    val_losses.append({"step": global_step, "loss": val_loss})
                    tqdm.write(f"  [Val] Step {global_step} | Val Loss: {val_loss:.4f}")
                    experiment.log_metric("val_loss", val_loss, step=global_step)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = os.path.join(args.output_dir, "best_model")
                        model.save_pretrained(save_path)
                        tqdm.write(f"  Saved best model (val_loss={val_loss:.4f})")

                # Environment evaluation
                if global_step % args.env_eval_interval == 0:
                    model.eval()
                    result = evaluate_in_env(
                        model, tokenizer, image_processor, cfg,
                        num_episodes=args.eval_episodes,
                        env_size=args.env_size,
                        max_steps_per_episode=args.max_steps_per_episode,
                        device=device,
                        output_format=args.output_format,
                    )
                    result["step"] = global_step
                    eval_results.append(result)
                    tqdm.write(f"  [Env] Step {global_step} | Success: {result['success_rate']:.2%} | "
                          f"Return: {result['avg_return']:.2f} | Steps: {result['avg_steps']:.1f}")
                    experiment.log_metrics({
                        "success_rate": result["success_rate"],
                        "avg_return": result["avg_return"],
                        "avg_steps": result["avg_steps"],
                    }, step=global_step)
                    model.train()

        tqdm.write(f"Epoch {epoch} complete | Avg Loss: {epoch_loss / max(num_batches, 1):.4f}")

    pbar.close()

    # Final save
    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)

    # Final evaluation
    model.eval()
    final_result = evaluate_in_env(
        model, tokenizer, image_processor, cfg,
        num_episodes=args.eval_episodes, env_size=args.env_size,
        max_steps_per_episode=args.max_steps_per_episode,
        device=device, output_format=args.output_format,
    )
    final_result["step"] = global_step
    eval_results.append(final_result)
    print(f"\nFinal Evaluation: Success={final_result['success_rate']:.2%} | "
          f"Return={final_result['avg_return']:.2f}")
    experiment.log_metrics({
        "final_success_rate": final_result["success_rate"],
        "final_avg_return": final_result["avg_return"],
        "final_avg_steps": final_result["avg_steps"],
    }, step=global_step)

    # Save training curves
    curves = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "eval_results": eval_results,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "training_curves.json"), "w") as f:
        json.dump(curves, f, indent=2)

    experiment.log_asset(os.path.join(args.output_dir, "training_curves.json"))
    experiment.end()
    print(f"\nTraining complete. Results saved to {args.output_dir}")
    return model


def validate(model, val_loader, device):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"]

            autocast_dtype = torch.bfloat16 if device.type in ["cuda", "cpu"] else torch.float16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            total_loss += loss.item()
            n += 1
            if n >= 50:  # Cap validation batches
                break
    model.train()
    return total_loss / max(n, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for VLA")
    # Model
    parser.add_argument("--checkpoint", type=str, default=None, help="Local model checkpoint path")
    parser.add_argument("--hf_model", type=str, default="lusxvr/nanoVLM-230M-8k", help="HuggingFace model ID")
    # Data
    parser.add_argument("--data_dir", type=str, default="vla/trajectories", help="Trajectory data directory")
    parser.add_argument("--output_format", type=str, default="action_only", choices=["action_only", "cot"])
    parser.add_argument("--max_length", type=int, default=512)
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr_mp", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    # Eval
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--env_eval_interval", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--max_steps_per_episode", type=int, default=20)
    parser.add_argument("--env_size", type=int, default=8)
    # Output
    parser.add_argument("--output_dir", type=str, default="vla/checkpoints/sft")
    # Comet ML
    parser.add_argument("--comet_project", type=str, default="nanoVLA", help="Comet ML project name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_sft(args)
