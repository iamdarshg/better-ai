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

def shrink_model_after_pruning(model: nn.Module):
    """
    Actually removes zeroed-out parameters and replaces layers with smaller ones.
    Only handles column-pruned Linear layers for now.
    """
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if layer has zeroed out columns
            weights = module.weight.data
            non_zero_mask = torch.sum(torch.abs(weights), dim=0) > 0

            if not torch.all(non_zero_mask):
                # Found zeroed columns, shrink!
                in_features = torch.sum(non_zero_mask).item()
                out_features = module.out_features

                new_layer = nn.Linear(in_features, out_features, bias=module.bias is not None)
                new_layer.weight.data = weights[:, non_zero_mask].clone()
                if module.bias is not None:
                    new_layer.bias.data = module.bias.data.clone()

                replacements.append((name, new_layer))

    for name, new_module in replacements:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            child_name = name

        setattr(parent, child_name, new_module)

def get_pruning_metrics(model: nn.Module) -> dict:
    """Calculates pruning statistics"""
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()

    return {
        "total_parameters": total_params,
        "zero_parameters": zero_params,
        "sparsity": zero_params / total_params if total_params > 0 else 0,
        "compression_ratio": total_params / (total_params - zero_params) if (total_params - zero_params) > 0 else float('inf')
    }
