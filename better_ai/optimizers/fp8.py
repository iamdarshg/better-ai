"""FP8 training infrastructure for DeepSeek model"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import warnings


class FP8Linear(nn.Module):
    """FP8 Linear layer with automatic quantization and scaling"""
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        use_fp8: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8 and self._check_fp8_support()
        
        if self.use_fp8:
            # Store weights in FP8 format
            self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn))
            # Scaling factor for quantization
            self.weight_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def _check_fp8_support(self) -> bool:
        """Check if FP8 is supported on current hardware"""
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to FP32")
            return False
        
        # Check CUDA version (FP8 requires CUDA 11.8+)
        try:
            major, minor = torch.version.cuda.split('.')[:2]
            if int(major) < 11 or (int(major) == 11 and int(minor) < 8):
                warnings.warn(f"CUDA {major}.{minor} does not support FP8, falling back to FP32")
                return False
        except:
            warnings.warn("Could not determine CUDA version, falling back to FP32")
            return False
        
        # Check if device supports FP8 (Ada Lovelace and newer)
        try:
            device_capability = torch.cuda.get_device_capability()
            major_sm, minor_sm = device_capability
            # FP8 support starts from compute capability 8.9 (Ada Lovelace)
            if (major_sm < 8) or (major_sm == 8 and minor_sm < 9):
                warnings.warn(f"Device compute capability {major_sm}.{minor_sm} does not support FP8, falling back to FP32")
                return False
        except:
            warnings.warn("Could not determine device capability, falling back to FP32")
            return False
        
        return True
    
    def reset_parameters(self):
        """Initialize parameters"""
        if self.use_fp8:
            # Initialize in FP32 then quantize
            weight_fp32 = torch.empty((self.out_features, self.in_features))
            nn.init.kaiming_uniform_(weight_fp32, a=math.sqrt(5))
            self.weight.data = weight_fp32.to(torch.float8_e4m3fn)
            self.weight_scale.data.fill_(1.0)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _quantize_fp8(self, tensor: torch.Tensor, scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to FP8 with scaling"""
        if scale is None:
            # Dynamic scaling based on absolute max
            scale = tensor.abs().max() / 448.0  # Max value for E4M3
            scale = torch.clamp(scale, min=1e-7)  # Avoid division by zero
        
        # Quantize to FP8
        quantized = (tensor / scale).to(torch.float8_e4m3fn)
        
        return quantized, scale
    
    def _dequantize_fp8(self, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize FP8 tensor back to higher precision"""
        return tensor.to(scale.dtype) * scale
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_fp8:
            # Quantize input to FP8
            input_fp8, input_scale = self._quantize_fp8(input)
            
            # Dequantize weight for computation
            weight_fp32 = self._dequantize_fp8(self.weight, self.weight_scale)
            
            # Compute FP8 GEMM in higher precision
            output = F.linear(input_fp8.to(weight_fp32.dtype), weight_fp32, self.bias)
            
            return output
        else:
            return F.linear(input, self.weight, self.bias)


class FP8Optimizer:
    """FP8 optimizer wrapper that handles scaling and quantization"""
    
    def __init__(
        self,
        params,
        optimizer_class: type,
        fp8_scaling_window: int = 16,
        fp8_delayed_scaling: bool = True,
        **optimizer_kwargs
    ):
        self.fp8_scaling_window = fp8_scaling_window
        self.fp8_delayed_scaling = fp8_delayed_scaling
        self.ama_history = {}
        
        # Initialize optimizer
        self.optimizer = optimizer_class(params, **optimizer_kwargs)
        
        # FP8 scaling factors
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.ama_history[p] = torch.zeros(fp8_scaling_window, device=p.device, dtype=torch.float32)
    
    def step(self, closure=None):
        """Perform optimization step with FP8 scaling"""
        
        # Update scaling factors if using delayed scaling
        if self.fp8_delayed_scaling:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        self._update_scaling_factor(p)
        
        # Perform optimization step
        self.optimizer.step(closure)
    
    def _update_scaling_factor(self, param: torch.Tensor):
        """Update scaling factor based on gradient statistics"""
        if param not in self.ama_history:
            return
        
        # Get gradient absolute max
        grad_ama = param.grad.abs().max().item()
        
        # Update history
        history = self.ama_history[param]
        history = torch.roll(history, -1)
        history[-1] = grad_ama
        self.ama_history[param] = history
        
        # Compute new scaling factor
        if self.fp8_delayed_scaling:
            # Use max of recent history
            new_scale = history.max().item()
        else:
            # Use current value
            new_scale = grad_ama
        
        # Clip scaling factor
        new_scale = max(new_scale, 1e-7)
        
        # Update parameter's scaling factor if it exists
        if hasattr(param, 'fp8_scale'):
            param.fp8_scale.data.fill_(new_scale)
    
    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients"""
        self.optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """Get optimizer state"""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state"""
        self.optimizer.load_state_dict(state_dict)


class FP8AdamW(FP8Optimizer):
    """FP8 version of AdamW optimizer"""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        fp8_scaling_window: int = 16,
        fp8_delayed_scaling: bool = True,
    ):
        super().__init__(
            params=params,
            optimizer_class=torch.optim.AdamW,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            fp8_scaling_window=fp8_scaling_window,
            fp8_delayed_scaling=fp8_delayed_scaling,
        )


class FP8LossScaler:
    """FP8 loss scaling for gradient overflow prevention"""
    
    def __init__(
        self,
        init_scale: float = 2.0**16,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2.0**24
    ):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self._iter = 0
        self._unskipped = 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient computation"""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients before clipping"""
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.div_(self.scale)
    
    def update_scale(self, overflow: bool = False):
        """Update scale based on overflow status"""
        if overflow:
            # Reduce scale on overflow
            self.scale = max(self.scale / self.scale_factor, self.min_scale)
            self._unskipped = 0
        else:
            # Increase scale periodically if no overflow
            self._unskipped += 1
            if self._unskipped >= self.scale_window:
                self.scale = min(self.scale * self.scale_factor, self.max_scale)
                self._unskipped = 0
        
        self._iter += 1
    
    def get_scale(self) -> float:
        """Get current scale value"""
        return self.scale


class FP8MixedPrecisionTrainer:
    """Mixed precision training with FP8 support"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: FP8Optimizer,
        loss_scaler: Optional[FP8LossScaler] = None,
        clip_grad_norm: float = 1.0,
        use_fp8: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler or FP8LossScaler()
        self.clip_grad_norm = clip_grad_norm
        self.use_fp8 = use_fp8 and self._check_fp8_availability()
        
        if self.use_fp8:
            self.model = self._convert_to_fp8(model)
    
    def _check_fp8_availability(self) -> bool:
        """Check if FP8 is available"""
        if not torch.cuda.is_available():
            return False
        
        try:
            major, minor = torch.version.cuda.split('.')[:2]
            if int(major) < 11 or (int(major) == 11 and int(minor) < 8):
                return False
        except:
            return False
        
        try:
            device_capability = torch.cuda.get_device_capability()
            major_sm, minor_sm = device_capability
            if (major_sm < 8) or (major_sm == 8 and minor_sm < 9):
                return False
        except:
            return False
        
        return True
    
    def _convert_to_fp8(self, model: nn.Module) -> nn.Module:
        """Convert linear layers to FP8"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with FP8 linear
                fp8_linear = FP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    use_fp8=self.use_fp8
                )
                
                # Copy weights
                with torch.no_grad():
                    fp8_linear.weight.copy_(module.weight)
                    if module.bias is not None:
                        fp8_linear.bias.copy_(module.bias)
                
                # Replace in model
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                parent = model
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                setattr(parent, module_name, fp8_linear)
        
        return model
    
    def training_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform one training step with FP8"""
        
        # Forward pass
        if self.use_fp8:
            # Scale loss for FP8 training
            with torch.cuda.amp.autocast(enabled=False):  # Disable AMP for FP8
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['last_hidden_state']
                
                # Compute loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                
                # Scale loss
                scaled_loss = self.loss_scaler.scale_loss(loss)
        else:
            # Standard training
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['last_hidden_state']
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            scaled_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # Backward pass
        scaled_loss.backward()
        
        # Unscale gradients
        if self.use_fp8:
            self.loss_scaler.unscale_gradients(self.optimizer.optimizer)
        
        # Gradient clipping
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        
        # Update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update loss scale
        if self.use_fp8:
            # Check for overflow
            overflow = any(torch.isinf(p.grad).any() or torch.isnan(p.grad).any() 
                          for p in self.model.parameters() if p.grad is not None)
            self.loss_scaler.update_scale(overflow)
        
        return loss.detach()


def get_fp8_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    use_fp8: bool = True,
    **kwargs
) -> FP8Optimizer:
    """Get FP8 optimizer for the model"""
    
    if use_fp8:
        return FP8AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )