from os.path import join
from typing import Dict, Union

import torch as th
import torch.nn as nn


def save_torch_model(
    models: Dict[str, Union[nn.Module, th.Tensor]],
    ckpt_dir: str,
    model_name: str = "models.pt",
) -> str:
    """Save [Pytorch] model to a pre-specified path
    Note: Currently, only th.Tensor and th.nn.Module are supported.
    """
    model_path = join(ckpt_dir, model_name)
    state_dicts = {}
    for name, model in models.items():
        if isinstance(model, th.Tensor):
            state_dicts[name] = {name: model}
        else:
            state_dicts[name] = model.state_dict()
    th.save(state_dicts, model_path)
    return model_path


def load_torch_model(models: Dict[str, Union[nn.Module, th.Tensor]], model_path: str):
    """Load [Pytorch] model from a pre-specified path"""
    state_dicts = th.load(model_path)
    for name, model in models.items():
        if isinstance(model, th.Tensor):
            models[name].copy_(state_dicts[name][name])
        else:
            model.load_state_dict(state_dicts[name])