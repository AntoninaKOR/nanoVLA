"""
Evaluation utilities for VLA models in MiniGrid EmptyEnv.

Provides:
- Environment rollout evaluation (success rate, return, steps)
- Training curve plotting
"""
import os
import json
import argparse
import sys

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.env_utils import make_env, get_obs_image, ACTION_NAMES, action_name_to_env_action


def predict_action(model, tokenizer, image_processor, cfg, obs_image,
                   device, output_format="action_only"):
    """
    Given an observation image, predict the next action using the VLA model.

    Returns:
        action_name (str): One of ACTION_NAMES, or None if parsing fails.
        raw_text (str): The raw model output text.
    """
    from data.processors import get_image_string

    processed_image, splitted_image_ratio = image_processor(obs_image)
    image_string = get_image_string(tokenizer, [splitted_image_ratio], cfg.mp_image_token_length)

    prompt = "You are a navigation agent in a grid world. Given the current observation, output the next action to reach the green goal square."
    user_content = image_string + prompt

    messages = [{"role": "user", "content": user_content}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer.encode(prompt_text, add_special_tokens=False)
    tokens = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    img_t = processed_image.to(device)

    max_new = 10 if output_format == "action_only" else 50
    gen_ids = model.generate(tokens, img_t, max_new_tokens=max_new, greedy=True)
    raw_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

    # Parse action from output
    action_name = parse_action(raw_text, output_format)
    return action_name, raw_text


def parse_action(text, output_format="action_only"):
    """Parse an action name from model output text."""
    text_lower = text.lower().strip()

    if output_format == "cot":
        # Look for "Action: <action_name>" pattern
        if "action:" in text_lower:
            after_action = text_lower.split("action:")[-1].strip()
            for action in ACTION_NAMES:
                if after_action.startswith(action):
                    return action

    # Direct match
    for action in ACTION_NAMES:
        if text_lower.startswith(action) or text_lower == action:
            return action

    # Fuzzy match
    for action in ACTION_NAMES:
        if action in text_lower:
            return action

    return None


def evaluate_in_env(model, tokenizer, image_processor, cfg,
                    num_episodes=50, env_size=8, max_steps_per_episode=100,
                    device=None, output_format="action_only"):
    """
    Evaluate VLA model by rolling out episodes in MiniGrid EmptyEnv.

    Returns dict with:
        success_rate, avg_return, avg_steps, episodes
    """
    if device is None:
        device = next(model.parameters()).device

    env = make_env(size=env_size)
    successes = 0
    total_returns = []
    total_steps_list = []
    parse_failures = 0

    model.eval()
    with torch.no_grad():
        for ep in range(num_episodes):
            obs, info = env.reset(seed=1000 + ep)
            episode_return = 0
            episode_steps = 0

            for step in range(max_steps_per_episode):
                obs_img = get_obs_image(obs)
                action_name, raw_text = predict_action(
                    model, tokenizer, image_processor, cfg,
                    obs_img, device, output_format
                )

                if action_name is None:
                    parse_failures += 1
                    action_name = "move_forward"  # Default fallback

                env_action = action_name_to_env_action(action_name)
                obs, reward, terminated, truncated, info = env.step(env_action)
                episode_return += reward
                episode_steps += 1

                if terminated or truncated:
                    break

            if terminated and reward > 0:
                successes += 1
            total_returns.append(episode_return)
            total_steps_list.append(episode_steps)

    env.close()

    result = {
        "success_rate": successes / num_episodes,
        "avg_return": float(np.mean(total_returns)),
        "avg_steps": float(np.mean(total_steps_list)),
        "parse_failure_rate": parse_failures / max(sum(total_steps_list), 1),
        "num_episodes": num_episodes,
    }
    return result


def plot_training_curves(curves_path, output_path=None):
    """Plot training curves from a training_curves.json file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    with open(curves_path) as f:
        curves = json.load(f)

    if output_path is None:
        output_path = os.path.dirname(curves_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training loss
    if curves.get("train_losses"):
        steps = [x["step"] for x in curves["train_losses"]]
        losses = [x["loss"] for x in curves["train_losses"]]
        axes[0].plot(steps, losses, alpha=0.5, label="train")
        # Smoothed
        if len(losses) > 10:
            window = min(20, len(losses) // 5)
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            axes[0].plot(steps[window - 1:], smoothed, label="train (smoothed)")
    if curves.get("val_losses"):
        steps = [x["step"] for x in curves["val_losses"]]
        losses = [x["loss"] for x in curves["val_losses"]]
        axes[0].plot(steps, losses, "o-", label="val")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Success rate
    if curves.get("eval_results"):
        steps = [x["step"] for x in curves["eval_results"]]
        sr = [x["success_rate"] for x in curves["eval_results"]]
        axes[1].plot(steps, sr, "s-", color="green")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Success Rate")
        axes[1].set_title("Environment Success Rate")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)

    # Average return
    if curves.get("eval_results"):
        ret = [x["avg_return"] for x in curves["eval_results"]]
        axes[2].plot(steps, ret, "d-", color="blue")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Average Return")
        axes[2].set_title("Average Episode Return")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_to = os.path.join(output_path, "training_curves.png")
    plt.savefig(save_to, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_to}")


def compare_experiments(experiment_dirs, labels=None, output_path="vla/comparison.png"):
    """Compare multiple experiments on the same plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    if labels is None:
        labels = [os.path.basename(d) for d in experiment_dirs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for exp_dir, label in zip(experiment_dirs, labels):
        curves_file = os.path.join(exp_dir, "training_curves.json")
        if not os.path.exists(curves_file):
            print(f"Skipping {exp_dir}: no training_curves.json")
            continue

        with open(curves_file) as f:
            curves = json.load(f)

        # Loss
        if curves.get("train_losses"):
            steps = [x["step"] for x in curves["train_losses"]]
            losses = [x["loss"] for x in curves["train_losses"]]
            if len(losses) > 10:
                window = min(20, len(losses) // 5)
                smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
                axes[0].plot(steps[window - 1:], smoothed, label=label)

        # Success rate
        if curves.get("eval_results"):
            steps = [x["step"] for x in curves["eval_results"]]
            sr = [x["success_rate"] for x in curves["eval_results"]]
            axes[1].plot(steps, sr, "o-", label=label)

        # Return
        if curves.get("eval_results"):
            ret = [x["avg_return"] for x in curves["eval_results"]]
            axes[2].plot(steps, ret, "s-", label=label)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Success Rate Comparison")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Average Return")
    axes[2].set_title("Return Comparison")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLA model or plot curves")
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "plot", "compare"])
    # Eval args
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hf_model", type=str, default="lusxvr/nanoVLM-230M-8k")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--env_size", type=int, default=8)
    parser.add_argument("--output_format", type=str, default="action_only")
    # Plot args
    parser.add_argument("--curves_path", type=str, default=None)
    # Compare args
    parser.add_argument("--experiment_dirs", nargs="+", default=None)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output_path", type=str, default="vla/comparison.png")
    args = parser.parse_args()

    if args.mode == "plot":
        if args.curves_path:
            plot_training_curves(args.curves_path)
    elif args.mode == "compare":
        if args.experiment_dirs:
            compare_experiments(args.experiment_dirs, args.labels, args.output_path)
    else:
        from models.vision_language_model import VisionLanguageModel
        from data.processors import get_tokenizer, get_image_processor

        source = args.checkpoint or args.hf_model
        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        model = VisionLanguageModel.from_pretrained(source).to(device)
        model.eval()
        cfg = model.cfg
        tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
        image_processor = get_image_processor(cfg.vit_img_size, cfg.vit_img_size, False)

        result = evaluate_in_env(
            model, tokenizer, image_processor, cfg,
            num_episodes=args.num_episodes,
            env_size=args.env_size,
            device=device,
            output_format=args.output_format,
        )
        print(json.dumps(result, indent=2))
