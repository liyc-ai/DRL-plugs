from os.path import join
from typing import Dict, Iterable, List, Tuple, Union

import torch as th
from numba import cuda
from stable_baselines3.common.torch_layers import create_mlp

# from GPUtil import showUtilization as gpu_usage
from torch import nn
from torch.optim import Optimizer

# --------------------- Setting --------------------


def set_torch(default_th_dtype: th.dtype = th.float32):
    th.set_default_dtype(default_th_dtype)
    th.utils.backcompat.broadcast_warning.enabled = True
    th.utils.backcompat.keepdim_warning.enabled = True


def clean_cuda():
    th.cuda.empty_cache()
    for gpu_id in range(th.cuda.device_count()):
        cuda.select_device(gpu_id)
        cuda.close()


# --------------------- Tensor ---------------------


def tensor2ndarray(tensors: Tuple[th.Tensor]):
    """Convert torch.Tensor to numpy.ndarray"""
    result = []
    for item in tensors:
        if th.is_tensor(item):
            result.append(item.detach().cpu().numpy())
        else:
            result.append(item)
    return result


# ------------------- Manipulate NN Module ----------------------


def move_device(modules: List[th.nn.Module], device: Union[str, th.device]):
    """Move net to specified device"""
    for module in modules:
        module.to(device)


def freeze_net(nets: List[nn.Module]):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = False


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


# ------------------ Initialization ----------------------------


def orthogonal_init_(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# ----------------------- Optimization ----------------------------


def gradient_descent(
    net_optim: Optimizer,
    loss: th.Tensor,
    parameters: Union[th.Tensor, Iterable[th.Tensor]] = None,
    max_grad_norm: float = None,
    retain_graph: bool = False,
):
    """Update network parameters with gradient descent."""
    net_optim.zero_grad()
    loss.backward(retain_graph=retain_graph)

    # gradient clip
    if all([parameters, max_grad_norm]):
        th.nn.utils.clip_grad_norm_(parameters, max_grad_norm)

    net_optim.step()
    return loss.item()


# ------------------------ Modules ------------------------


def variable(shape: Tuple[int, ...]):
    return nn.Parameter(th.zeros(shape), requires_grad=True)


def mlp(
    input_shape: Tuple[int,],
    output_shape: Tuple[int,],
    net_arch: List[int],
    activation_fn: nn.Module = nn.ReLU,
    squash_output: bool = False,
) -> Tuple[List[nn.Module], int]:
    """
    :return: (net, feature_dim)
    """
    # output feature dimension
    if output_shape[0] == -1:
        if len(net_arch) > 0:
            feature_shape = (net_arch[-1], 0)
        else:
            raise ValueError("Empty MLP!")
    else:
        feature_shape = output_shape
    # networks
    net = nn.Sequential(
        *create_mlp(
            input_shape[0], output_shape[0], net_arch, activation_fn, squash_output
        )
    )
    return net, feature_shape


def cnn(
    input_shape: List[int],
    output_dim: int,
    net_arch: List[Tuple[int]],
    activation_fn: nn.Module = nn.ReLU,
) -> Tuple[List[nn.Module], int]:
    """
    :param input_shape: (channel, ...)
    :net_arch: list of conv2d, i.e., (output_channel, kernel_size, stride, padding)
    """
    input_channel = input_shape[0]

    if len(net_arch) > 0:
        module = [nn.Conv2d(input_channel, *net_arch[0]), activation_fn()]
    else:
        raise ValueError("Empty CNN!")

    # parse modules
    for i in range(1, len(net_arch)):
        module.append(nn.Conv2d(net_arch[i - 1][0], *net_arch[i]))
        module.append(activation_fn())
    net = nn.Sequential(*module)
    net.add_module("flatten-0", nn.Flatten())

    # Compute shape by doing one forward pass
    with th.no_grad():
        n_flatten = net(th.randn(input_shape).unsqueeze(dim=0)).shape[1]

    # We use -1 to just extract the feature
    if output_dim == -1:
        return net, n_flatten
    else:
        net.add_module("linear-0", nn.Linear(n_flatten, output_dim))
        return net, output_dim
