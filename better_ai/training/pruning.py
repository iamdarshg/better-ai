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
    Handles both column (input) and row (output) pruned Linear layers.
    """
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weights = module.weight.data

            # 1. Check for zeroed out columns (input features)
            col_mask = torch.sum(torch.abs(weights), dim=0) > 0

            # 2. Check for zeroed out rows (output features)
            row_mask = torch.sum(torch.abs(weights), dim=1) > 0

            if not torch.all(col_mask) or not torch.all(row_mask):
                # Found zeroed parameters, shrink!
                in_features = torch.sum(col_mask).item()
                out_features = torch.sum(row_mask).item()

                new_layer = nn.Linear(in_features, out_features, bias=module.bias is not None)

                # Filter weights by both masks
                temp_weights = weights[row_mask, :]
                new_weights = temp_weights[:, col_mask]
                new_layer.weight.data = new_weights.clone()

                if module.bias is not None:
                    new_layer.bias.data = module.bias.data[row_mask].clone()

                replacements.append((name, new_layer))

    for name, new_module in replacements:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            child_name = name

        setattr(parent, child_name, new_module)

def prune_attention_heads(model: nn.Module, heads_to_prune: dict):
    """
    Physically removes attention heads from the model.

    Args:
        model: The model to prune (should be a DeepSeekModel or similar)
        heads_to_prune: Dictionary mapping layer index to list of head indices to prune
    """
    for layer_idx, heads in heads_to_prune.items():
        if not hasattr(model, 'layers') or layer_idx >= len(model.layers):
            continue

        layer = model.layers[layer_idx]
        if not hasattr(layer, 'self_attn'):
            continue

        attn = layer.self_attn
        if not hasattr(attn, 'q_proj'):
            continue

        device = attn.q_proj.weight.device
        head_dim = attn.head_dim
        # Get current number of heads from weight shape to be robust to repeated pruning
        current_q_out = attn.q_proj.weight.data.shape[0]
        num_heads = current_q_out // head_dim

        # Create mask for heads to keep
        keep_heads = [i for i in range(num_heads) if i not in heads]
        if not keep_heads:
            continue # Cannot prune all heads

        # 1. Prune q_proj
        q_weights = attn.q_proj.weight.data.view(num_heads, head_dim, -1)
        attn.q_proj.weight.data = q_weights[keep_heads].view(-1, q_weights.shape[-1]).clone()
        attn.q_proj.out_features = len(keep_heads) * head_dim

        # 2. Prune o_proj (columns)
        o_weights = attn.o_proj.weight.data.view(-1, num_heads, head_dim)
        attn.o_proj.weight.data = o_weights[:, keep_heads].view(o_weights.shape[0], -1).clone()
        attn.o_proj.in_features = len(keep_heads) * head_dim

        # Update num_heads
        attn.num_heads = len(keep_heads)

        # Handle GQA if applicable
        if hasattr(attn, 'num_key_value_heads'):
            current_kv_out = attn.k_proj.weight.data.shape[0]
            num_kv_heads = current_kv_out // head_dim

            # num_groups should be based on the relationship between ORIGINAL Q and KV heads
            # But if we don't have that, we can try to infer it.
            # Usually num_heads is a multiple of num_kv_heads.
            num_groups = num_heads // num_kv_heads

            if num_groups > 0:
                # Find which KV heads are still needed
                kv_heads_needed = set()
                for h in keep_heads:
                    kv_heads_needed.add(h // num_groups)

                keep_kv_heads = sorted(list(kv_heads_needed))

                # Prune k_proj
                k_weights = attn.k_proj.weight.data.view(num_kv_heads, head_dim, -1)
                attn.k_proj.weight.data = k_weights[keep_kv_heads].view(-1, k_weights.shape[-1]).clone()
                attn.k_proj.out_features = len(keep_kv_heads) * head_dim

                # Prune v_proj
                v_weights = attn.v_proj.weight.data.view(num_kv_heads, head_dim, -1)
                attn.v_proj.weight.data = v_weights[keep_kv_heads].view(-1, v_weights.shape[-1]).clone()
                attn.v_proj.out_features = len(keep_kv_heads) * head_dim

                attn.num_key_value_heads = len(keep_kv_heads)
                attn.num_key_value_groups = attn.num_heads // attn.num_key_value_heads

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
