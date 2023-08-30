import os
from os.path import join
from typing import Dict, Union

import torch as th
import torch.nn as nn

from mllogger.helper import copys


def archive(src_log_dir: str, exp_name: str, tgt_dir: str = "archived"):
    """Archive logs named [exp_name] in [src_dir] to tgt_dir"""
    os.makedirs(tgt_dir, exist_ok=True)
    src_dir = join(src_log_dir, exp_name)
    assert os.path.exists(src_dir)
    copys(src_dir, join(tgt_dir, exp_name))


def save_model(
    models: Dict[str, Union[nn.Module, th.Tensor]],
    ckpt_dir: str,
    model_name: str = "models.pt",
) -> str:
    """Save model to pre-specified path
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


def load_model(models: Dict[str, Union[nn.Module, th.Tensor]], model_path: str):
    """Load model from pre-specified path"""
    state_dicts = th.load(model_path)
    for name, model in models.items():
        if isinstance(model, th.Tensor):
            models[name].copy_(state_dicts[name][name])
        else:
            model.load_state_dict(state_dicts[name])
