"""
8-bit AdamW optimizer implementation
Uses block-wise quantization to reduce optimizer memory by 75%
Optimizer states stored in 8-bit instead of FP32
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class AdamW8bit(torch.optim.Optimizer):
    """
    8-bit AdamW optimizer with block-wise quantization.

    Reduces memory usage by ~75% compared to standard AdamW:
    - Standard AdamW: 2 FP32 states = 8 bytes per parameter
    - 8-bit AdamW: 2 8-bit states + 2 FP32 block scales = ~2 bytes per parameter

    Based on Dettmers et al. "8-bit Optimizers via Block-wise Quantization"
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        block_size: int = 2048,
    ):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Coefficients for running averages (beta1, beta2)
            eps: Term added for numerical stability
            weight_decay: Weight decay coefficient
            block_size: Block size for quantization (default 2048)
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            block_size=block_size,
        )
        super().__init__(params, defaults)

        # Initialize 8-bit state storage
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["step"] = 0

                    # Initialize moments in 8-bit
                    numel = p.numel()
                    num_blocks = (numel + block_size - 1) // block_size

                    # Exp avg (momentum) - stored in 8-bit
                    state["exp_avg"] = torch.zeros(
                        numel, dtype=torch.uint8, device=p.device
                    )
                    state["exp_avg_scale"] = torch.zeros(
                        num_blocks, dtype=torch.float32, device=p.device
                    )

                    # Exp avg sq (variance) - stored in 8-bit
                    state["exp_avg_sq"] = torch.zeros(
                        numel, dtype=torch.uint8, device=p.device
                    )
                    state["exp_avg_sq_scale"] = torch.zeros(
                        num_blocks, dtype=torch.float32, device=p.device
                    )

    def _quantize_block(
        self, tensor: torch.Tensor, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to 8-bit with block-wise scaling.

        Returns:
            quantized: 8-bit quantized tensor
            scale: FP32 scale per block
        """
        numel = tensor.numel()
        num_blocks = (numel + block_size - 1) // block_size

        # Pad if necessary
        if numel % block_size != 0:
            padding = block_size - (numel % block_size)
            tensor = torch.cat(
                [
                    tensor.flatten(),
                    torch.zeros(padding, device=tensor.device, dtype=tensor.dtype),
                ]
            )

        # Reshape into blocks
        blocks = tensor.view(num_blocks, block_size)

        # Compute scale per block (max absolute value)
        abs_max = blocks.abs().max(dim=1)[0]
        scale = abs_max / 127.0  # 127 is max for int8
        scale = torch.clamp(scale, min=1e-10)  # Avoid division by zero

        # Quantize
        quantized = (
            (blocks / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        )

        # Flatten and remove padding
        quantized = quantized.view(-1)[:numel]

        return quantized.to(torch.uint8), scale

    def _dequantize_block(
        self, quantized: torch.Tensor, scale: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """
        Dequantize 8-bit tensor back to FP32.
        """
        numel = quantized.numel()
        num_blocks = (numel + block_size - 1) // block_size

        # Pad if necessary
        if numel % block_size != 0:
            padding = block_size - (numel % block_size)
            quantized = torch.cat(
                [
                    quantized.flatten(),
                    torch.zeros(padding, device=quantized.device, dtype=torch.uint8),
                ]
            )

        # Reshape into blocks
        blocks = quantized.view(num_blocks, block_size).to(torch.float32)

        # Dequantize
        dequantized = blocks * scale.unsqueeze(1)

        # Flatten and remove padding
        return dequantized.view(-1)[:numel]

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            block_size = group["block_size"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Increment step
                state["step"] += 1
                step = state["step"]

                # Apply weight decay
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # Dequantize momentum
                exp_avg = self._dequantize_block(
                    state["exp_avg"], state["exp_avg_scale"], block_size
                ).view_as(p)

                # Dequantize variance
                exp_avg_sq = self._dequantize_block(
                    state["exp_avg_sq"], state["exp_avg_sq_scale"], block_size
                ).view_as(p)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Compute step size
                step_size = lr / bias_correction1

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Re-quantize and store
                state["exp_avg"][:], state["exp_avg_scale"] = self._quantize_block(
                    exp_avg.flatten(), block_size
                )
                state["exp_avg_sq"][:], state["exp_avg_sq_scale"] = (
                    self._quantize_block(exp_avg_sq.flatten(), block_size)
                )

        return loss


class AdamW8bitAvailable:
    """Check if 8-bit optimizer dependencies are available"""

    @staticmethod
    def check() -> bool:
        """Check if bitsandbytes is available"""
        try:
            import bitsandbytes as bnb

            return True
        except ImportError:
            return False


def get_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    use_8bit: bool = True,
    use_fp8: bool = False,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Get optimizer with memory-efficient options.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        use_8bit: Use 8-bit AdamW (saves 75% optimizer memory)
        use_fp8: Use FP8 optimizer wrapper
        **kwargs: Additional optimizer arguments

    Returns:
        Configured optimizer
    """
    if use_8bit and AdamW8bitAvailable.check():
        # Use bitsandbytes 8-bit AdamW if available
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
        )
    elif use_8bit:
        # Fallback to custom 8-bit implementation
        return AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif use_fp8:
        # Use FP8 optimizer wrapper
        from .fp8 import FP8AdamW

        return FP8AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        # Standard AdamW
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
        )


# Convenience function for backward compatibility
def get_fp8_optimizer(*args, **kwargs):
    """Backward compatibility - use get_optimizer instead"""
    return get_optimizer(*args, **kwargs)
