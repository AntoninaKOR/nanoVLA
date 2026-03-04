import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from vla.env_utils import make_env, get_obs_image, action_name_to_env_action
from vla.expert import get_expert_actions


def collect_trajectories(num_episodes=5000, env_size=8, output_dir="vla/trajectories", seed=42):
    """
    Collect expert trajectories and save as image files + JSON metadata.

    Each trajectory step produces:
      - An image file: {episode_id}_{step_id}.png
      - A metadata entry: {episode_id, step_id, action, done}

    Args:
        num_episodes: Number of episodes to collect.
        env_size: Size of the EmptyEnv grid.
        output_dir: Directory to save trajectories.
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(size=env_size)
    all_steps = []
    total_steps = 0
    success_count = 0

    for ep_idx in range(num_episodes):
        obs, info = env.reset(seed=seed + ep_idx)
        expert_actions = get_expert_actions(env)

        if expert_actions is None:
            continue

        for step_idx, action_name in enumerate(expert_actions):
            # Save current observation
            img = get_obs_image(obs)
            img_filename = f"{ep_idx:05d}_{step_idx:03d}.png"
            img.save(images_dir / img_filename)

            step_record = {
                "episode_id": ep_idx,
                "step_id": step_idx,
                "image_path": f"images/{img_filename}",
                "action": action_name,
                "total_steps_in_episode": len(expert_actions),
            }
            all_steps.append(step_record)

            # Take the action
            env_action = action_name_to_env_action(action_name)
            obs, reward, terminated, truncated, info = env.step(env_action)
            total_steps += 1

        if terminated:
            success_count += 1

        if (ep_idx + 1) % 100 == 0:
            print(f"Collected {ep_idx + 1}/{num_episodes} episodes, {total_steps} total steps")

        # Periodic save every 1000 episodes to avoid data loss
        if (ep_idx + 1) % 1000 == 0:
            with open(output_path / "steps.json", "w") as f:
                json.dump(all_steps, f)
            print(f"  Checkpoint saved at episode {ep_idx + 1}")

    env.close()

    # Save metadata
    metadata = {
        "num_episodes": num_episodes,
        "env_size": env_size,
        "total_steps": total_steps,
        "success_rate": success_count / num_episodes if num_episodes > 0 else 0,
        "action_space": ["turn_left", "turn_right", "move_forward"],
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_path / "steps.json", "w") as f:
        json.dump(all_steps, f)

    print(f"\nCollection complete:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Success rate: {metadata['success_rate']:.2%}")
    print(f"  Saved to: {output_path}")

    return all_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect expert trajectories for VLA training")
    parser.add_argument("--num_episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--env_size", type=int, default=8, help="MiniGrid EmptyEnv size")
    parser.add_argument("--output_dir", type=str, default="vla/trajectories", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    collect_trajectories(
        num_episodes=args.num_episodes,
        env_size=args.env_size,
        output_dir=args.output_dir,
        seed=args.seed,
    )
