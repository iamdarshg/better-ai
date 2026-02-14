"""Core transformer components for DeepSeek model"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Any
from .rope import RoPECache
from .features.recursive_scratchpad import RecursiveScratchpad
from .features.cot_specialization import CoTSpecializationHeads
from .features.inner_monologue import InnerMonologue
from .features.star_module import STaRModule
from .features.tool_use import ToolUseHeads
from .features.specialized_head import SpecializedHead
from .features.gbnf_constraint import GBNFConstraint
from .features.json_enforcer import JSONEnforcer
from .features.entropic_steering import EntropicSteering
from .tidar import TiDAR
from .reward_model import BranchRewardModel, MultiAttributeRewardModel, HierarchicalRewardModel
from .generation import generate, compute_loss, self_correct
from .features.reasoning_rewards import TraceValidityScorer, StructuralSignalReward, AHAMomentDetector


# To avoid circular imports
_MoELayer = None

class RMSNorm(nn.Module):
    """RMS Normalization layer"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with support for GQA"""
    
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int, head_dim: int, dropout: float = 0.0, use_nope: bool = False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.use_nope = use_nope
        
        if head_dim * num_heads != hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)

        if not self.use_nope:
            self.rope_cache = RoPECache(
                dim=self.head_dim,
                max_seq_len=4096,
                base=10000,
                device=torch.device("cpu")
            )
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, num_heads: int):
        """Reshape tensor for attention computation"""
        return tensor.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)
    
    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads to match query heads"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = self._shape(query_states, q_len, bsz, self.num_heads)
        key_states = self._shape(key_states, q_len, bsz, self.num_key_value_heads)
        value_states = self._shape(value_states, q_len, bsz, self.num_key_value_heads)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if not self.use_nope:
            self.rope_cache.to(query_states.device)
            query_states, key_states = self.rope_cache(query_states, key_states)
        
        # Handle key-value caching for inference
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        if use_cache:
            present = (key_states, value_states)
        else:
            present = None
        
        # Handle GQA (Grouped Query Attention)
        if self.num_key_value_groups > 1:
            key_states = self.repeat_kv(key_states, self.num_key_value_groups)
            value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights += attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, present


class TransformerBlock(nn.Module):
    """Transformer block with RMSNorm and SwiGLU or MoE"""
    
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int, head_dim: int, 
                 intermediate_size: int, norm_eps: float = 1e-6, dropout: float = 0.0,
                 use_moe: bool = False, num_experts: int = 8, num_experts_per_token: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        
        # Feed-forward
        if use_moe:
            global _MoELayer
            if _MoELayer is None:
                from .moe import MoELayer
                _MoELayer = MoELayer
            self.mlp = _MoELayer(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                expert_intermediate_size=intermediate_size,
                dropout=dropout
            )
        else:
            self.mlp = SwiGLU(hidden_size, intermediate_size)
        
        # Normalization
        self.input_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        
        # Dropout
        self.residual_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = self.residual_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        aux_loss = None
        # Check for MoE layer without repeated imports
        if _MoELayer is not None and isinstance(self.mlp, _MoELayer):
            hidden_states, aux_loss, _ = self.mlp(hidden_states, attention_mask=attention_mask)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.residual_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        
        # Add aux_loss if it exists
        outputs += (aux_loss,)

        return outputs


class DeepSeekModel(nn.Module):
    """DeepSeek-inspired Transformer model"""
    
    def __init__(self, config, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        self.hidden_size = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_length = config.max_seq_length
        self.device_str = str(device) if device else None
        
        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, self.hidden_size, self.padding_idx)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            use_moe = (i % getattr(config, "use_moe_every_n_layers", 2) == 0) and i > 0
            self.layers.append(
                TransformerBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    intermediate_size=config.intermediate_dim,
                    norm_eps=config.norm_eps,
                    dropout=config.residual_dropout,
                    use_moe=use_moe,
                    num_experts=getattr(config, "num_experts", 8),
                    num_experts_per_token=getattr(config, "num_experts_per_token", 2)
                )
            )
        
        # Final normalization
        self.norm = RMSNorm(self.hidden_size, eps=self.config.norm_eps)

        # Language model head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Advanced features initialization
        self._init_advanced_features(config, device)

        # Replace attention layers if requested
        if getattr(config, "use_striped_attention", False):
            self._replace_with_striped_attention(config, device)
        elif getattr(config, "use_linear_attention", False):
            self._replace_with_linear_attention(config, device)
        
        # Initialize weights
        self.apply(self._init_weights)


    def _init_advanced_features(self, config, device):
        """Initialize all advanced features if enabled in config"""
        if getattr(config, "use_recursive_scratchpad", False):
            self.scratchpad = RecursiveScratchpad(
                config.hidden_dim,
                max_iterations=getattr(config, "scratchpad_max_iterations", 8),
                scratchpad_dim=getattr(config, "scratchpad_hidden_dim", 32),
            )

        if getattr(config, "use_tidar", False):
            self.tidar = TiDAR(
                hidden_dim=config.hidden_dim,
                num_steps=getattr(config, "tidar_num_steps", 5),
                diffusion_dim=getattr(config, "tidar_diffusion_dim", 128),
                num_layers=getattr(config, "tidar_num_layers", 2)
            )

        if getattr(config, "use_cot_specialization", False):
            self.cot_heads = CoTSpecializationHeads(
                config.hidden_dim,
                num_cot_heads=getattr(config, "cot_num_heads", 5),
                cot_hidden_dim=getattr(config, "cot_hidden_dim", 32),
            )

        if getattr(config, "use_inner_monologue", False):
            self.inner_monologue = InnerMonologue(
                config.hidden_dim,
                private_subspace_dim=getattr(config, "private_subspace_dim", 4096),
            )

        if getattr(config, "use_star", False):
            self.star = STaRModule(
                config.hidden_dim,
                num_bootstrap_rounds=getattr(config, "star_bootstrap_rounds", 3),
                consistency_samples=getattr(config, "star_consistency_samples", 10),
            )

        if getattr(config, "use_tool_heads", False):
            self.tool_heads = ToolUseHeads(
                config.hidden_dim,
                tool_vocab_size=getattr(config, "tool_vocab_size", 32),
                tool_hidden_dim=getattr(config, "tool_hidden_dim", 32),
            )

        if getattr(config, "use_json_db_ops_head", False):
            self.json_db_ops_head = SpecializedHead(
                hidden_dim=config.hidden_dim,
                internal_dim=getattr(config, "json_db_ops_internal_dim", 256),
                ratio=getattr(config, "json_db_ops_ratio", 0.1)
            )

        if getattr(config, "use_math_reasoning_head", False):
            self.math_reasoning_head = SpecializedHead(
                config.hidden_dim,
                internal_dim=getattr(config, "math_reasoning_internal_dim", 256),
                ratio=getattr(config, "math_reasoning_ratio", 0.1)
            )

        if getattr(config, "use_algorithm_head", False):
            self.algorithm_head = SpecializedHead(
                config.hidden_dim,
                internal_dim=getattr(config, "algorithm_internal_dim", 256),
                ratio=getattr(config, "algorithm_ratio", 0.1)
            )

        if getattr(config, "use_grammar_constraints", False):
            self.gbnf_constraint = GBNFConstraint(config.hidden_dim, grammar_type=getattr(config, "grammar_type", "gbnf"))

        if getattr(config, "enforce_json_output", False):
            self.json_enforcer = JSONEnforcer(config.hidden_dim)

        if getattr(config, "use_entropic_steering", False):
            self.entropic_steering = EntropicSteering(config.hidden_dim, entropy_threshold=getattr(config, "entropy_threshold", 2.5))

        # Reward models and other heads (Only initialized if explicitly requested or in production)
        self.reward_model = BranchRewardModel(config, hidden_dim=512)
        if getattr(config, "use_reward_models", False):
            self.multi_attr_reward = MultiAttributeRewardModel(config, num_attributes=7, num_quantiles=5)
            self.hrm = HierarchicalRewardModel(config)

        if getattr(config, "use_reasoning_rewards", False):
            self.trace_validity_scorer = TraceValidityScorer(self)
            self.structural_reward_engine = StructuralSignalReward()
            self.aha_moment_detector = AHAMomentDetector()

        if getattr(config, "use_value_head", False):
            self.value_head = nn.Linear(config.hidden_dim, 1, bias=False)

        # Cache for weight entropy to avoid redundant compute
        self._cached_weight_entropy = 0.0
        self._last_entropy_compute_step = -1
    
    def calculate_weight_entropy(self) -> float:
        """Calculate average entropy of weight distributions across linear layers."""
        total_entropy = 0.0
        count = 0
        try:
            for name, param in self.named_parameters():
                if "weight" in name and param.dim() >= 2 and param.numel() > 100:
                    w = param.detach().float()
                    # Standardize to get a distribution
                    w_min, w_max = w.min(), w.max()
                    hist = torch.histc(w, bins=50, min=float(w_min), max=float(w_max))
                    prob = hist / (hist.sum() + 1e-10)
                    entropy = -(prob * torch.log(prob + 1e-10)).sum()
                    total_entropy += entropy.item()
                    count += 1
        except Exception:
            return 0.0
        return total_entropy / count if count > 0 else 0.0

    def _init_weights(self, module):
        """Initialize weights using scaled normal distribution"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings matrix of the model if new_num_tokens != config.vocab_size."""
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return

        old_embeddings = self.get_input_embeddings()
        new_embeddings = nn.Embedding(new_num_tokens, self.config.hidden_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # numbers of tokens to copy
        n = min(old_embeddings.weight.shape[0], new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        self.set_input_embeddings(new_embeddings)

        # Also resize lm_head
        old_lm_head = self.lm_head
        self.lm_head = nn.Linear(self.config.hidden_dim, new_num_tokens, bias=False)
        self.lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
        self.lm_head.weight.data[:n, :] = old_lm_head.weight.data[:n, :]

        self.config.vocab_size = new_num_tokens
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_advanced_features: bool = False,
        use_compiled_mask: bool = False,
    ) -> Any:
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device)
        
        hidden_states = inputs_embeds
        
        # Prepare attention mask for the layers
        raw_attention_mask = attention_mask
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                # Convert to causal mask with bounds checking
                if seq_length <= 0:
                    raise ValueError(f"Invalid sequence length: {seq_length}")
                
                # Ensure device consistency
                device = inputs_embeds.device
                causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=device)).bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask.unsqueeze(0)
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            else:
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None
        total_aux_loss = torch.tensor(0.0, device=inputs_embeds.device)

        for i, (layer_module, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            # Map outputs based on their existence
            # 0: hidden_states
            # 1: self_attn_weights (optional)
            # 2: present_key_value (optional)
            # 3: aux_loss (optional)
            
            curr_idx = 1
            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[curr_idx],)
                curr_idx += 1

            if use_cache:
                next_cache = next_cache + (layer_outputs[curr_idx],)
                curr_idx += 1

            if layer_outputs[-1] is not None:
                total_aux_loss += layer_outputs[-1]
        
        hidden_states = self.norm(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Compute advanced features if requested
        advanced_features = {}
        if return_advanced_features:
            advanced_features = self._compute_advanced_features(
                hidden_states, input_ids, raw_attention_mask,
                logits=logits, use_compiled_mask=use_compiled_mask
            )

        if not return_dict:
            res = [logits, hidden_states, next_cache, all_hidden_states, all_self_attns]
            if return_advanced_features:
                res.append(advanced_features)
            return tuple(v for v in res if v is not None)
        
        result = {
            "logits": logits,
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
            "aux_loss": total_aux_loss,
        }
        if return_advanced_features:
            result["advanced_features"] = advanced_features

        return result

    def _replace_with_linear_attention(self, config: Any, device: Optional[torch.device] = None):
        """Replace standard attention with Linear Attention."""
        for layer in self.layers:
            linear_attn = LinearAttention(
                hidden_size=config.hidden_dim,
                num_heads=config.num_attention_heads,
            )
            if device:
                linear_attn.to(device)
            layer.self_attn = linear_attn

    def _replace_with_striped_attention(self, config: Any, device: Optional[torch.device] = None):
        """Replace standard attention with Striped Attention"""
        from .striped_attention import StripedAttention

        for i, layer in enumerate(self.layers):
            # Create Striped Attention module
            striped_attn = StripedAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_attention_heads,
                num_key_value_heads=getattr(config, "num_key_value_heads", config.num_attention_heads // 2),
                striped_block_size=getattr(config, "striped_block_size", 1024),
                dropout=getattr(config, "attention_dropout", 0.0),
                rope_theta=getattr(config, "rope_theta", 10000.0),
                max_seq_len=getattr(config, "max_seq_length", 524288),
            )

            # Move to device if specified
            if device is not None:
                striped_attn = striped_attn.to(device)

            # Replace attention in layer
            layer.self_attn = striped_attn

    def _compute_advanced_features(self, hidden_states, input_ids, attention_mask, logits=None, use_compiled_mask: bool = False):
        """Compute all advanced features and return them in a dictionary"""
        advanced_outputs = {}

        # Ensure hidden_states is on correct device if needed
        # (Usually already handled, but being safe)

        # Recursive Scratchpad
        if hasattr(self, "scratchpad") and getattr(self.config, "use_recursive_scratchpad", False):
            scratchpad_out = self.scratchpad(hidden_states)
            advanced_outputs["scratchpad"] = scratchpad_out
            hidden_states = scratchpad_out["scratchpad_output"]

        # TiDAR
        if hasattr(self, "tidar"):
            prompt_repr = hidden_states.mean(dim=1)
            tidar_out = self.tidar(hidden_states, prompt_repr)
            advanced_outputs["tidar"] = tidar_out
            hidden_states = tidar_out["refined_scratchpad"]

        # CoT Specialization
        if hasattr(self, "cot_heads"):
            cot_out = self.cot_heads(hidden_states, is_reasoning_phase=True)
            advanced_outputs["cot"] = cot_out
            hidden_states = cot_out["final_output"]

        # Inner Monologue
        if hasattr(self, "inner_monologue"):
            monologue_out = self.inner_monologue(
                hidden_states,
                token_ids=input_ids,
                thought_token_id=getattr(self.config, "thought_token_id", None),
            )
            advanced_outputs["inner_monologue"] = monologue_out
            hidden_states = monologue_out["output"]

        # STaR
        if hasattr(self, "star"):
            star_out = self.star(hidden_states, [hidden_states])
            advanced_outputs["star"] = star_out

        # Tool-Use Heads
        if hasattr(self, "tool_heads"):
            tool_out = self.tool_heads(hidden_states)
            advanced_outputs["tool_use"] = tool_out

        # Specialized Heads
        specialized_outputs = []
        if hasattr(self, "json_db_ops_head"):
            json_db_ops_out = self.json_db_ops_head(hidden_states)
            specialized_outputs.append(json_db_ops_out * self.json_db_ops_head.ratio)
            advanced_outputs["json_db_ops_head"] = json_db_ops_out

        if hasattr(self, "math_reasoning_head"):
            math_reasoning_out = self.math_reasoning_head(hidden_states)
            specialized_outputs.append(math_reasoning_out * self.math_reasoning_head.ratio)
            advanced_outputs["math_reasoning_head"] = math_reasoning_out

        if hasattr(self, "algorithm_head"):
            algorithm_out = self.algorithm_head(hidden_states)
            specialized_outputs.append(algorithm_out * self.algorithm_head.ratio)
            advanced_outputs["algorithm_head"] = algorithm_out

        if specialized_outputs:
            hidden_states = hidden_states + sum(specialized_outputs)

        # Grammar Constraints
        if hasattr(self, "gbnf_constraint") and logits is not None:
            gbnf_out = self.gbnf_constraint(hidden_states, logits, input_ids=input_ids, use_compiled_mask=use_compiled_mask)
            advanced_outputs["gbnf"] = gbnf_out
            logits = gbnf_out["constrained_logits"]

        # JSON Enforcement
        if hasattr(self, "json_enforcer") and logits is not None:
            json_out = self.json_enforcer(hidden_states, logits, input_ids)
            advanced_outputs["json"] = json_out
            logits = json_out["constrained_logits"]

        # Entropic Steering
        if hasattr(self, "entropic_steering") and logits is not None:
            # Optionally use cached weight entropy or compute if needed
            # For performance, we might not want to compute every forward pass
            # But for the purpose of this feature, we'll use the cached one if available
            weight_entropy = getattr(self, "_cached_weight_entropy", 0.0)
            entropy_out = self.entropic_steering(hidden_states, logits, weight_entropy=weight_entropy)
            advanced_outputs["entropic_steering"] = entropy_out

        # Update logits in advanced_outputs if they were changed
        advanced_outputs["constrained_logits"] = logits

        # Reward models
        if hasattr(self, "reward_model"):
            advanced_outputs["reward"] = self.reward_model(hidden_states, attention_mask)
        if hasattr(self, "multi_attr_reward"):
            advanced_outputs["multi_attr_reward"] = self.multi_attr_reward(hidden_states, attention_mask)

        # Reasoning-specific rewards
        if hasattr(self, "trace_validity_scorer"):
            # We need the full decoded text for some of these
            # This is a simplification; in production, we'd pass the actual traces
            full_text = ""
            if input_ids is not None and hasattr(self, "tokenizer") and self.tokenizer:
                full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

            advanced_outputs["trace_validity"] = self.trace_validity_scorer.score_trace([full_text], "Solve the problem")
            advanced_outputs["structural_signal"] = self.structural_reward_engine.compute_reward(full_text)
            advanced_outputs["aha_moment"] = self.aha_moment_detector.compute_aha_reward(full_text)

        # Value head output
        if hasattr(self, "value_head"):
            advanced_outputs["value"] = self.value_head(hidden_states)

        return advanced_outputs

    generate = generate
    compute_loss = compute_loss
    self_correct = self_correct


class LinearAttention(nn.Module):
    """Gated Linear Attention (GLA) variant."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.g_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = hidden_states
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        g = torch.sigmoid(self.g_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Simple linear attention: Q * (K^T * V)
        kv = torch.einsum("b h s d, b h s e -> b h d e", k, v)
        output = torch.einsum("b h s d, b h d e -> b h s e", q, kv)

        output = self.o_proj((output * g).transpose(1, 2).reshape(batch_size, seq_len, -1))
        return output, None, None