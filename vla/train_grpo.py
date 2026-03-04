"""
GRPO (Guided Regularized Policy Optimization) training for VLA.

Implements GRPO where the model directly outputs actions.
Initializes from SFT checkpoint and optimizes via online RL in MiniGrid EmptyEnv.

Reference: https://arxiv.org/abs/2402.03300
"""
import os
import sys
import math
import json
import time
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from comet_ml import Experiment
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

from vla.env_utils import make_env, get_obs_image, ACTION_NAMES, action_name_to_env_action
from vla.evaluate import evaluate_in_env, parse_action

torch.manual_seed(42)


def collect_rollouts(model, ref_model, tokenizer, image_processor, cfg,
                     num_episodes, env_size, device, output_format,
                     num_samples_per_state=4, max_steps=100):
    """
    Collect rollout data from the current policy for GRPO.

    For each state encountered, we sample `num_samples_per_state` actions
    from the policy and score them based on episode outcome.

    Returns list of experience dicts.
    """
    env = make_env(size=env_size)
    experiences = []

    model.eval()
    ref_model.eval()

    with torch.no_grad():
        for ep in range(num_episodes):
            obs, info = env.reset(seed=2000 + ep)
            episode_transitions = []

            for step in range(max_steps):
                obs_img = get_obs_image(obs)
                processed_image, splitted_image_ratio = image_processor(obs_img)
                image_string = get_image_string(
                    tokenizer, [splitted_image_ratio], cfg.mp_image_token_length
                )

                prompt = "You are a navigation agent in a grid world. Given the current observation, output the next action to reach the green goal square."
                user_content = image_string + prompt
                messages = [{"role": "user", "content": user_content}]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                encoded = tokenizer.encode(prompt_text, add_special_tokens=False)
                tokens = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
                img_t = processed_image.to(device)

                # Sample multiple actions from policy
                sampled_actions = []
                sampled_texts = []
                sampled_log_probs = []
                ref_log_probs = []

                max_new = 10 if output_format == "action_only" else 50

                for _ in range(num_samples_per_state):
                    gen_ids = model.generate(
                        tokens, img_t, max_new_tokens=max_new,
                        top_k=50, top_p=0.95, temperature=0.7, greedy=False
                    )
                    raw_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
                    action = parse_action(raw_text, output_format)
                    sampled_actions.append(action)
                    sampled_texts.append(raw_text)

                    # Compute log probability of generated sequence
                    full_ids = torch.cat([tokens[0], gen_ids[0]], dim=0).unsqueeze(0)
                    attn_mask = torch.ones_like(full_ids)

                    # Policy log prob
                    logits, _ = model(full_ids, img_t.unsqueeze(0) if img_t.ndim == 3 else img_t,
                                      attention_mask=attn_mask)
                    logits = model.decoder.head(logits) if not model.decoder.lm_use_tokens else logits
                    prompt_len = tokens.size(1)
                    gen_logits = logits[0, prompt_len - 1:-1]  # Logits predicting gen tokens
                    gen_targets = full_ids[0, prompt_len:]
                    log_p = F.log_softmax(gen_logits, dim=-1)
                    token_log_probs = log_p.gather(1, gen_targets.unsqueeze(1)).squeeze(1)
                    sampled_log_probs.append(token_log_probs.sum().item())

                    # Reference log prob
                    ref_logits, _ = ref_model(full_ids, img_t.unsqueeze(0) if img_t.ndim == 3 else img_t,
                                              attention_mask=attn_mask)
                    ref_logits = ref_model.decoder.head(ref_logits) if not ref_model.decoder.lm_use_tokens else ref_logits
                    ref_gen_logits = ref_logits[0, prompt_len - 1:-1]
                    ref_log_p = F.log_softmax(ref_gen_logits, dim=-1)
                    ref_token_log_probs = ref_log_p.gather(1, gen_targets.unsqueeze(1)).squeeze(1)
                    ref_log_probs.append(ref_token_log_probs.sum().item())

                # Pick best action (majority vote or first valid)
                valid_actions = [a for a in sampled_actions if a is not None]
                if valid_actions:
                    chosen_action = valid_actions[0]
                else:
                    chosen_action = "move_forward"

                episode_transitions.append({
                    "tokens": tokens.cpu(),
                    "image": img_t.cpu(),
                    "sampled_actions": sampled_actions,
                    "sampled_texts": sampled_texts,
                    "sampled_log_probs": sampled_log_probs,
                    "ref_log_probs": ref_log_probs,
                    "chosen_action": chosen_action,
                })

                env_action = action_name_to_env_action(chosen_action)
                obs, reward, terminated, truncated, info = env.step(env_action)

                if terminated or truncated:
                    break

            # Compute rewards for all samples per state
            episode_success = terminated and reward > 0
            episode_reward = 1.0 if episode_success else -0.1

            for trans in episode_transitions:
                sample_rewards = []
                for action in trans["sampled_actions"]:
                    if action is None:
                        sample_rewards.append(-1.0)  # Invalid action penalty
                    elif episode_success:
                        sample_rewards.append(1.0)
                    else:
                        sample_rewards.append(-0.1)

                # GRPO: normalize rewards within the group
                r = np.array(sample_rewards, dtype=np.float32)
                if r.std() > 1e-8:
                    r = (r - r.mean()) / r.std()
                else:
                    r = r - r.mean()

                trans["normalized_rewards"] = r.tolist()
                experiences.append(trans)

    env.close()
    return experiences


def grpo_update(model, ref_model, experiences, optimizer, device, tokenizer,
                kl_coeff=0.1, clip_range=0.2, max_grad_norm=1.0):
    """
    Perform a GRPO policy update step.

    For each sampled response, we compute:
        loss = -advantage * min(ratio, clip(ratio)) + kl_coeff * KL(policy || ref)
    """
    model.train()
    total_loss = 0
    num_updates = 0

    for exp in experiences:
        tokens = exp["tokens"].to(device)
        img_t = exp["image"].to(device)
        rewards = exp["normalized_rewards"]
        old_log_probs = exp["sampled_log_probs"]
        ref_lps = exp["ref_log_probs"]

        for i, (text, old_lp, ref_lp, reward) in enumerate(
            zip(exp["sampled_texts"], old_log_probs, ref_lps, rewards)
        ):
            action = exp["sampled_actions"][i]
            if action is None:
                continue

            # Re-tokenize the response
            response_ids = torch.tensor(
                tokenizer.encode(text, add_special_tokens=False), dtype=torch.long
            ).to(device)

            if len(response_ids) == 0:
                continue

            full_ids = torch.cat([tokens[0], response_ids], dim=0).unsqueeze(0)
            attn_mask = torch.ones_like(full_ids)

            # Current policy log prob
            autocast_dtype = torch.bfloat16 if device.type in ["cuda", "cpu"] else torch.float16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits, _ = model(full_ids, img_t.unsqueeze(0) if img_t.ndim == 3 else img_t,
                                  attention_mask=attn_mask)
                logits = model.decoder.head(logits) if not model.decoder.lm_use_tokens else logits

                prompt_len = tokens.size(1)
                gen_logits = logits[0, prompt_len - 1:-1]
                gen_targets = full_ids[0, prompt_len:]
                log_p = F.log_softmax(gen_logits, dim=-1)
                new_log_prob = log_p.gather(1, gen_targets.unsqueeze(1)).squeeze(1).sum()

                # Policy ratio
                ratio = torch.exp(new_log_prob - old_lp)
                clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

                advantage = torch.tensor(reward, device=device, dtype=torch.float32)

                # Clipped surrogate objective
                surr1 = ratio * advantage
                surr2 = clipped_ratio * advantage
                policy_loss = -torch.min(surr1, surr2)

                # KL penalty (towards reference)
                kl = new_log_prob - ref_lp
                kl_loss = kl_coeff * kl

                loss = policy_loss + kl_loss

            loss.backward()
            num_updates += 1
            total_loss += loss.item()

            if num_updates % 4 == 0:  # Mini-batch update
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

    # Final update for remaining gradients
    if num_updates % 4 != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(num_updates, 1)


def train_grpo(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Load SFT model as starting point
    print(f"Loading SFT model from: {args.sft_checkpoint}")
    model = VisionLanguageModel.from_pretrained(args.sft_checkpoint).to(device)
    cfg = model.cfg

    # Create frozen reference model
    ref_model = VisionLanguageModel.from_pretrained(args.sft_checkpoint).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    global tokenizer
    tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    image_processor = get_image_processor(cfg.vit_img_size, cfg.vit_img_size, False)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    os.makedirs(args.output_dir, exist_ok=True)
    eval_results = []
    train_losses = []

    # Comet ML
    experiment = Experiment(project_name=args.comet_project, auto_metric_logging=False)
    experiment.set_name(f"grpo_{args.output_format}")
    experiment.log_parameters(vars(args))

    print(f"\nStarting GRPO training (direct action)")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Episodes per iteration: {args.episodes_per_iter}")
    print(f"  Samples per state: {args.samples_per_state}")
    print(f"  KL coeff: {args.kl_coeff}")

    for iteration in tqdm(range(1, args.num_iterations + 1), desc="GRPO Training", unit="iter"):
        tqdm.write(f"\n--- Iteration {iteration}/{args.num_iterations} ---")

        # Collect rollouts
        experiences = collect_rollouts(
            model, ref_model, tokenizer, image_processor, cfg,
            num_episodes=args.episodes_per_iter,
            env_size=args.env_size,
            device=device,
            output_format=args.output_format,
            num_samples_per_state=args.samples_per_state,
        )
        tqdm.write(f"  Collected {len(experiences)} state experiences")

        # GRPO update
        loss = grpo_update(
            model, ref_model, experiences, optimizer, device, tokenizer,
            kl_coeff=args.kl_coeff,
            clip_range=args.clip_range,
        )
        train_losses.append({"iteration": iteration, "loss": loss})
        tqdm.write(f"  GRPO Loss: {loss:.4f}")
        experiment.log_metric("grpo_loss", loss, step=iteration)

        # Evaluate
        if iteration % args.eval_interval == 0:
            result = evaluate_in_env(
                model, tokenizer, image_processor, cfg,
                num_episodes=args.eval_episodes,
                env_size=args.env_size,
                max_steps_per_episode=args.max_steps_per_episode,
                device=device,
                output_format=args.output_format,
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

            # Save checkpoint
            save_path = os.path.join(args.output_dir, f"iter_{iteration}")
            model.save_pretrained(save_path)

    # Final save
    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)

    # Final eval
    result = evaluate_in_env(
        model, tokenizer, image_processor, cfg,
        num_episodes=args.eval_episodes, env_size=args.env_size,
        max_steps_per_episode=args.max_steps_per_episode,
        device=device, output_format=args.output_format,
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
    print(f"GRPO training complete. Results saved to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training - direct action")
    parser.add_argument("--sft_checkpoint", type=str, required=True, help="Path to SFT model")
    parser.add_argument("--output_format", type=str, default="action_only")
    parser.add_argument("--env_size", type=int, default=8)
    # GRPO hyperparams
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--episodes_per_iter", type=int, default=10)
    parser.add_argument("--samples_per_state", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl_coeff", type=float, default=0.1)
    parser.add_argument("--clip_range", type=float, default=0.2)
    # Eval
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--max_steps_per_episode", type=int, default=20)
    # Output
    parser.add_argument("--output_dir", type=str, default="vla/checkpoints/grpo_action")
    # Comet ML
    parser.add_argument("--comet_project", type=str, default="nanoVLA", help="Comet ML project name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_grpo(args)
