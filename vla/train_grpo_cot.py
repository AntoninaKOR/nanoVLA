"""
GRPO training with chain-of-thought (CoT) output format.

The model generates a brief description of the current state and a plan,
followed by the action. Format:
    "Step X of Y. Action: <action_name>"

This script is nearly identical to train_grpo.py but uses the 'cot' output format,
which encourages the model to reason about its observations before acting.
"""
import os
import sys
import json
import argparse

import torch
import torch.optim as optim
from comet_ml import Experiment
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor

from vla.evaluate import evaluate_in_env
from vla.train_grpo import collect_rollouts, grpo_update

torch.manual_seed(42)

# Needed as global for grpo_update
tokenizer = None


def train_grpo_cot(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Load SFT model (trained with cot format)
    print(f"Loading SFT model from: {args.sft_checkpoint}")
    model = VisionLanguageModel.from_pretrained(args.sft_checkpoint).to(device)
    cfg = model.cfg

    # Frozen reference model
    ref_model = VisionLanguageModel.from_pretrained(args.sft_checkpoint).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    global tokenizer
    tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    image_processor = get_image_processor(cfg.vit_img_size, cfg.vit_img_size, False)

    # Set the global tokenizer in train_grpo module too
    import vla.train_grpo as grpo_module
    grpo_module.tokenizer = tokenizer

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    os.makedirs(args.output_dir, exist_ok=True)
    eval_results = []
    train_losses = []

    # Comet ML
    experiment = Experiment(project_name=args.comet_project, auto_metric_logging=False)
    experiment.set_name("grpo_cot")
    experiment.log_parameters(vars(args))

    print(f"\nStarting GRPO training (text + action / CoT)")
    print(f"  Output format: cot")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Episodes per iteration: {args.episodes_per_iter}")

    for iteration in tqdm(range(1, args.num_iterations + 1), desc="GRPO-CoT Training", unit="iter"):
        tqdm.write(f"\n--- Iteration {iteration}/{args.num_iterations} ---")

        experiences = collect_rollouts(
            model, ref_model, tokenizer, image_processor, cfg,
            num_episodes=args.episodes_per_iter,
            env_size=args.env_size,
            device=device,
            output_format="cot",
            num_samples_per_state=args.samples_per_state,
        )
        tqdm.write(f"  Collected {len(experiences)} state experiences")

        loss = grpo_update(
            model, ref_model, experiences, optimizer, device, tokenizer,
            kl_coeff=args.kl_coeff,
            clip_range=args.clip_range,
        )
        train_losses.append({"iteration": iteration, "loss": loss})
        tqdm.write(f"  GRPO Loss: {loss:.4f}")
        experiment.log_metric("grpo_loss", loss, step=iteration)

        if iteration % args.eval_interval == 0:
            result = evaluate_in_env(
                model, tokenizer, image_processor, cfg,
                num_episodes=args.eval_episodes,
                env_size=args.env_size,
                max_steps_per_episode=args.max_steps_per_episode,
                device=device,
                output_format="cot",
            )
            result["iteration"] = iteration
            eval_results.append(result)
            tqdm.write(f"  [Eval] Success: {result['success_rate']:.2%} | "
                  f"Return: {result['avg_return']:.2f} | Steps: {result['avg_steps']:.1f}")
            experiment.log_metrics({
                "success_rate": result["success_rate"],
                "avg_return": result["avg_return"],
                "avg_steps": result["avg_steps"],
            }, step=iteration)

            save_path = os.path.join(args.output_dir, f"iter_{iteration}")
            model.save_pretrained(save_path)

    # Final
    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)

    result = evaluate_in_env(
        model, tokenizer, image_processor, cfg,
        num_episodes=args.eval_episodes, env_size=args.env_size,
        max_steps_per_episode=args.max_steps_per_episode,
        device=device, output_format="cot",
    )
    result["iteration"] = args.num_iterations
    eval_results.append(result)
    print(f"\nFinal: Success={result['success_rate']:.2%} | Return={result['avg_return']:.2f}")
    experiment.log_metrics({
        "final_success_rate": result["success_rate"],
        "final_avg_return": result["avg_return"],
        "final_avg_steps": result["avg_steps"],
    }, step=args.num_iterations)

    curves = {
        "train_losses": train_losses,
        "eval_results": eval_results,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "training_curves.json"), "w") as f:
        json.dump(curves, f, indent=2)

    experiment.log_asset(os.path.join(args.output_dir, "training_curves.json"))
    experiment.end()
    print(f"GRPO CoT training complete. Results saved to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training - text + action (CoT)")
    parser.add_argument("--sft_checkpoint", type=str, required=True, help="Path to SFT model (cot format)")
    parser.add_argument("--env_size", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--episodes_per_iter", type=int, default=10)
    parser.add_argument("--samples_per_state", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl_coeff", type=float, default=0.1)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--max_steps_per_episode", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="vla/checkpoints/grpo_cot")
    # Comet ML
    parser.add_argument("--comet_project", type=str, default="nanoVLA", help="Comet ML project name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_grpo_cot(args)
