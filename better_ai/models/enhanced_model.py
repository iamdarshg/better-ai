"""
Enhanced DeepSeek Model with all advanced features integrated
Includes Ring Attention, BR-RM, CoT, Inner Monologue, STaR, Tool-Use, GBNF, JSON, Entropic Steering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from .core import DeepSeekModel, TransformerBlock, LinearAttention
from .ring_attention import RingAttention
from .reward_model import BranchRewardModel, MultiAttributeRewardModel
from .advanced_features import (
    RecursiveScratchpad,
    CoTSpecializationHeads,
    InnerMonologue,
    STaRModule,
    ToolUseHeads,
    GBNFConstraint,
    JSONEnforcer,
    EntropicSteering,
)
from ..config import ModelConfig
from .generation import generate, compute_loss, self_correct


class EnhancedDeepSeekModel(nn.Module):
    """
    Enhanced DeepSeek Model with all advanced features
    Integrates Ring Attention, RLHF, reasoning, and specialized heads
    """
    
    def __init__(self, config: ModelConfig, device: Optional[torch.device] = None):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.device_str = str(device) if device else None
        
        # Core model
        self.model = DeepSeekModel(config)
        
        # Move core model to device if specified
        if device is not None:
            self.model = self.model.to(device)
        
        # Replace attention layers with Ring Attention or Linear Attention if enabled
        if config.use_ring_attention:
            self._replace_with_ring_attention(config, device)
        elif config.use_linear_attention:
            self._replace_with_linear_attention(config, device)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False, device=device)
        
        # Advanced features
        if config.use_recursive_scratchpad:
            self.scratchpad = RecursiveScratchpad(
                config.hidden_dim,
                max_iterations=config.scratchpad_max_iterations,
                scratchpad_dim=config.scratchpad_hidden_dim,
            )
        
        if config.use_cot_specialization:
            self.cot_heads = CoTSpecializationHeads(
                config.hidden_dim,
                num_cot_heads=config.cot_num_heads,
                cot_hidden_dim=config.cot_hidden_dim,
            )
        
        if config.use_inner_monologue:
            self.inner_monologue = InnerMonologue(
                config.hidden_dim,
                private_subspace_dim=config.private_subspace_dim,
            )
        
        if config.use_star:
            self.star = STaRModule(
                config.hidden_dim,
                num_bootstrap_rounds=config.star_bootstrap_rounds,
                consistency_samples=config.star_consistency_samples,
            )
        
        if config.use_tool_heads:
            self.tool_heads = ToolUseHeads(
                config.hidden_dim,
                tool_vocab_size=config.tool_vocab_size,
                tool_hidden_dim=config.tool_hidden_dim,
            )
        
        if config.use_grammar_constraints:
            self.gbnf_constraint = GBNFConstraint(config.hidden_dim, grammar_type=config.grammar_type)
        
        if config.enforce_json_output:
            self.json_enforcer = JSONEnforcer(config.hidden_dim)
        
        if config.use_entropic_steering:
            self.entropic_steering = EntropicSteering(config.hidden_dim, entropy_threshold=config.entropy_threshold)
        
        # Reward models
        self.reward_model = BranchRewardModel(config, hidden_dim=512)
        self.multi_attr_reward = MultiAttributeRewardModel(config, num_attributes=5, num_quantiles=5)

        # Value head for PPO
        self.value_head = nn.Linear(config.hidden_dim, 1, bias=False, device=device)

    generate = generate
    compute_loss = compute_loss
    self_correct = self_correct

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize the token embeddings."""
        self.model.resize_token_embeddings(new_num_tokens)
        self.lm_head = nn.Linear(self.config.hidden_dim, new_num_tokens, bias=False, device=self.device_str)

    def _replace_with_linear_attention(self, config: ModelConfig, device: Optional[torch.device] = None):
        """Replace standard attention with Linear Attention."""
        for layer in self.model.layers:
            linear_attn = LinearAttention(
                hidden_size=config.hidden_dim,
                num_heads=config.num_attention_heads,
            )
            if device:
                linear_attn.to(device)
            layer.self_attn = linear_attn
    
    def _replace_with_ring_attention(self, config: ModelConfig, device: Optional[torch.device] = None):
        """Replace standard attention with Ring Attention"""
        for i, layer in enumerate(self.model.layers):
            # Create Ring Attention module
            ring_attn = RingAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                block_size=config.ring_block_size,
                dropout=config.attention_dropout,
                rope_theta=config.rope_theta,
                max_seq_len=config.max_seq_length,
            )
            
            # Move to device if specified
            if device is not None:
                ring_attn = ring_attn.to(device)
            
            # Replace attention in layer
            layer.self_attn = ring_attn
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_advanced_features: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with all advanced features
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            return_advanced_features: If True, return outputs from advanced features
        
        Returns:
            Dictionary with logits and optionally advanced feature outputs
        """
        # Core model forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        
        hidden_states = outputs["last_hidden_state"]  # (batch_size, seq_len, hidden_dim)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)  # (batch_size, seq_len, vocab_size)
        
        result = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        if not return_advanced_features:
            return result
        
        # Advanced features
        advanced_outputs = {}
        
        # Recursive Scratchpad
        if self.config.use_recursive_scratchpad:
            scratchpad_out = self.scratchpad(hidden_states)
            advanced_outputs["scratchpad"] = scratchpad_out
            hidden_states = scratchpad_out["scratchpad_output"]
        
        # CoT Specialization
        if self.config.use_cot_specialization:
            cot_out = self.cot_heads(hidden_states, is_reasoning_phase=True)
            advanced_outputs["cot"] = cot_out
            hidden_states = cot_out["final_output"]
        
        # Inner Monologue
        if self.config.use_inner_monologue:
            monologue_out = self.inner_monologue(
                hidden_states,
                token_ids=input_ids,
                thought_token_id=self.config.thought_token_id,
            )
            advanced_outputs["inner_monologue"] = monologue_out
            hidden_states = monologue_out["output"]
        
        # STaR (Self-Taught Reasoner)
        if self.config.use_star:
            # Collect reasoning traces from previous steps
            reasoning_traces = [hidden_states]  # Placeholder
            star_out = self.star(hidden_states, reasoning_traces)
            advanced_outputs["star"] = star_out
        
        # Tool-Use Heads
        if self.config.use_tool_heads:
            tool_out = self.tool_heads(hidden_states)
            advanced_outputs["tool_use"] = tool_out
        
        # Grammar Constraints
        if self.config.use_grammar_constraints:
            gbnf_out = self.gbnf_constraint(hidden_states, logits)
            advanced_outputs["gbnf"] = gbnf_out
            logits = gbnf_out["constrained_logits"]
        
        # JSON Enforcement
        if self.config.enforce_json_output:
            json_out = self.json_enforcer(hidden_states, logits, input_ids)
            advanced_outputs["json"] = json_out
            logits = json_out["constrained_logits"]
        
        # Entropic Steering
        if self.config.use_entropic_steering:
            entropy_out = self.entropic_steering(hidden_states, logits)
            advanced_outputs["entropic_steering"] = entropy_out
        
        # Reward models (for RLHF)
        reward_score = self.reward_model(hidden_states, attention_mask)
        multi_attr_reward = self.multi_attr_reward(hidden_states, attention_mask)
        
        advanced_outputs["reward"] = reward_score
        advanced_outputs["multi_attr_reward"] = multi_attr_reward
        
        # Value head output
        value = self.value_head(hidden_states)
        advanced_outputs["value"] = value

        result["advanced_features"] = advanced_outputs
        result["logits"] = logits
        
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens using the model
        
        Args:
            input_ids: (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-K sampling parameter
            top_p: Nucleus (Top-P) sampling parameter
            use_cache: Whether to use KV cache
        
        Returns:
            Generated token ids (batch_size, seq_len + max_new_tokens)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize past key values
        past_key_values = None
        
        # Store all generated tokens
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=generated[:, -1:] if past_key_values else generated,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_advanced_features=False,
            )
            
            logits = outputs["logits"][:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-K sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)
            
            # Top-P (Nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold (nucleus filtering)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # scatter sorted tensors to original indexing
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch_size,)
            
            # Append to generated
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Stop if EOS token
            if (next_tokens == 2).all():  # Assuming EOS token ID is 2
                break
        
        return generated
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reward_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for training
        
        Args:
            input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len) - language modeling targets
            reward_labels: (batch_size,) - reward/preference labels for RLHF
        
        Returns:
            Dictionary with loss components
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_advanced_features=True,
        )
        
        logits = outputs["logits"]
        
        # Language modeling loss
        lm_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            reduction="mean",
        )
        
        losses = {"lm_loss": lm_loss}
        
        # Reward modeling loss (if labels provided)
        if reward_labels is not None:
            advanced_features = outputs.get("advanced_features", {})
            
            # BR-RM loss
            if "reward" in advanced_features:
                reward_pred = advanced_features["reward"]
                reward_loss = F.mse_loss(reward_pred, reward_labels.float())
                losses["reward_loss"] = reward_loss
            
            # Multi-attribute reward loss
            if "multi_attr_reward" in advanced_features:
                multi_attr = advanced_features["multi_attr_reward"]
                # Simplified multi-attribute loss
                multi_loss = sum([
                    F.mse_loss(v.mean(dim=-1), reward_labels)
                    for v in multi_attr.values()
                    if isinstance(v, torch.Tensor) and v.dim() > 1
                ]) / max(len([v for v in multi_attr.values() if isinstance(v, torch.Tensor)]), 1)
                losses["multi_attr_loss"] = multi_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss
        
        return losses

