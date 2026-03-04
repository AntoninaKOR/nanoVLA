"""
MiniGrid environment utilities for VLA training.
Handles environment creation, observation rendering, and action space.
"""
import gymnasium as gym
import numpy as np
from PIL import Image
from minigrid.wrappers import RGBImgObsWrapper


# MiniGrid EmptyEnv action space (subset we use)
ACTION_NAMES = ["turn_left", "turn_right", "move_forward"]
ACTION_TO_ID = {name: i for i, name in enumerate(ACTION_NAMES)}
ID_TO_ACTION = {i: name for i, name in enumerate(ACTION_NAMES)}

# MiniGrid native action mapping
MINIGRID_ACTION_MAP = {
    "turn_left": 0,
    "turn_right": 1,
    "move_forward": 2,
}


def make_env(size=8, render_mode="rgb_array"):
    """Create a MiniGrid EmptyEnv with RGB observation wrapper."""
    env = gym.make(f"MiniGrid-Empty-{size}x{size}-v0", render_mode=render_mode)
    env = RGBImgObsWrapper(env)  # Get pixel observations
    return env


def get_obs_image(obs) -> Image.Image:
    """Extract PIL Image from MiniGrid observation dict."""
    img_array = obs["image"]
    return Image.fromarray(img_array.astype(np.uint8))


def action_name_to_env_action(action_name: str) -> int:
    """Convert our action name to MiniGrid action integer."""
    return MINIGRID_ACTION_MAP[action_name]
