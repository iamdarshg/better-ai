"""
Width pruning for expert layers in MoE models.
"""

import torch
import torch.nn as nn

def prune_expert_widths(model: nn.Module, pruning_ratio: float, expert_layer_names: list):
    """
    Prunes the widths of the expert layers in a model.

    Args:
        model: The model to prune.
        pruning_ratio: The fraction of weights to prune.
        expert_layer_names: A list of names of the expert layers to prune.
    """
    for name, module in model.named_modules():
        if name in expert_layer_names:
            if isinstance(module, nn.Linear):
                l1_norm = torch.norm(module.weight, p=1, dim=0)
                num_to_prune = int(pruning_ratio * l1_norm.shape[0])

                if num_to_prune > 0:
                    threshold = torch.kthvalue(l1_norm, num_to_prune).values
                    mask = l1_norm <= threshold
                    module.weight.data[:, mask] = 0
