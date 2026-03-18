"""Shared network utilities: weight initialization and normalization layer factory.

Follows the pix2pix/CUT convention of Gaussian weight initialization (mean=0, std=0.02).
"""

import functools
from typing import Callable, Type

import torch.nn as nn


def get_norm_layer(norm_type: str = "instance") -> Callable:
    """Return a normalization layer constructor by name.

    Args:
        norm_type: One of ``"instance"``, ``"batch"``, or ``"none"``.

    Returns:
        A callable that accepts a channel count and returns an ``nn.Module``.

    Raises:
        ValueError: If ``norm_type`` is not recognized.
    """
    if norm_type == "instance":
        # affine=False matches original CUT/pix2pix convention
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "none":
        # Return a no-op layer; lambda must accept the channel arg
        return lambda channels: nn.Identity()
    else:
        raise ValueError(
            f"Unknown norm_type '{norm_type}'. Choose from: 'instance', 'batch', 'none'."
        )


def init_weights(
    net: nn.Module,
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> None:
    """Initialize network weights in-place.

    Applies Gaussian (``"normal"``) or alternative initialization to Conv and
    Linear layers, and constant initialization to BatchNorm2d scale/bias.

    Args:
        net: The network whose weights will be initialized.
        init_type: One of ``"normal"``, ``"xavier"``, ``"kaiming"``,
            ``"orthogonal"``.
        init_gain: Scaling factor / std for the chosen scheme.

    Raises:
        ValueError: If ``init_type`` is not recognized.
    """

    def _init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        is_conv_or_linear = hasattr(m, "weight") and (
            "Conv" in classname or "Linear" in classname
        )
        if is_conv_or_linear:
            if init_type == "normal":
                nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise ValueError(
                    f"Unknown init_type '{init_type}'. Choose from: "
                    "'normal', 'xavier', 'kaiming', 'orthogonal'."
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            # scale → 1, bias → 0 (standard)
            nn.init.normal_(m.weight.data, mean=1.0, std=init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(_init_func)
