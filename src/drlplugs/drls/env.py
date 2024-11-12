from typing import Any, Dict, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


def _get_space_info(obj: gym.Space) -> Tuple[Tuple[int, ...], str]:
    if isinstance(obj, Box):
        shape = obj.shape
        type_ = "float"
    elif isinstance(obj, Discrete):
        shape = (obj.n.item(),)
        type_ = "int"
    else:
        raise TypeError("Currently only Box and Discrete are supported!")
    return shape, type_


def get_env_info(env: gym.Env) -> Dict[str, Union[Tuple[int, ...], str]]:
    state_shape, _ = _get_space_info(env.observation_space)
    action_shape, action_dtype = _get_space_info(env.action_space)

    env_info = {
        "state_shape": state_shape,
        "action_shape": action_shape,
        "action_dtype": action_dtype,
    }

    if isinstance(env.action_space, Box):
        env_info["action_scale"] = float(env.action_space.high[0])

    return env_info


def make_env(env_id: str) -> gym.Env:
    """Currently we only support the below simple env style"""
    try:
        env = gym.make(env_id)
    except:
        raise ValueError("Unsupported env id!")
    return env


def reset_env_fn(env: gym.Env, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    next_state, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return (next_state, info)
