import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .init import init_weight


def create_decomposed_weights(regularizer, input_dim, output_dim, num_transforms, num_bases=-1):
    assert regularizer in ["none", "basis", "bdd", "diag", "scalar"]
    if num_bases <= 0:
        regularizer = "none"

    weights = dict()
    if regularizer == "none":
        weight = nn.Parameter(th.Tensor(num_transforms, input_dim * output_dim))
        weights = {"weight": weight}
    elif regularizer == "basis":
        w_comp = nn.Parameter(th.Tensor(num_transforms, num_bases))
        weight = nn.Parameter(th.Tensor(num_bases, input_dim * output_dim))
        weights = {"w_comp": w_comp, "weight": weight}
    elif regularizer == "bdd":
        if input_dim % num_bases != 0 or output_dim % num_bases != 0:
            raise ValueError("Feature size must be a multiplier of num_bases (%d)." % num_bases)
        # assuming input_dim and output_dim are both divisible by num_bases
        weight = nn.Parameter(th.Tensor(num_transforms, input_dim * output_dim // num_bases))
        weights = {"weight": weight}
    elif regularizer == "diag":
        if input_dim != output_dim:
            raise ValueError("Input size must equal to output size.")
        w_comp = nn.Parameter(th.Tensor(num_transforms, num_bases))
        weight = nn.Parameter(th.Tensor(num_bases, input_dim))
        weights = {"w_comp": w_comp, "weight": weight}
    elif regularizer == "scalar":
        if input_dim != output_dim:
            raise ValueError("Input size must equal to output size.")
        w_comp = nn.Parameter(th.Tensor(num_transforms, num_bases))
        weight = nn.Parameter(th.Tensor(num_bases, 1))
        weights = {"w_comp": w_comp, "weight": weight}

    for w in weights.values():
        init_weight(w, init="uniform")

    return weights
