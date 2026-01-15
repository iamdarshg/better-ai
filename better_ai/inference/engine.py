"""Inference engine for DeepSeek model with optimizations"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Union, Tuple, Generator
import time
import json
from dataclasses import asdict
import logging

from ..config import InferenceConfig
from ..models.core import DeepSeekModel
from ..models.moe import DeepSeekMoEModel
from ..utils import get_device


class InferenceEngine:
    """Optimized inference engine for DeepSeek models"""
    
    def __init__(
        self,
        model: Union[DeepSeekModel, DeepSeekMoEModel],
        config: InferenceConfig,
        tokenizer=None
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # Device setup
        self.device = get_device()
        self.model.to(self.device)
        
        # KV cache management
        self.kv_cache = None
        self.max_cache_size = config.cache_size or 1000
        
        # Generation parameters
        self.max_new_tokens = config.max_new_tokens
        self.do_sample = config.do_sample
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.repetition_penalty = config.repetition_penalty
        
        # Optimization flags
        self.use_kv_cache = config.use_kv_cache
        self.use_fp8_inference = config.use_fp8_inference
        self.batch_size = config.batch_size
        self.streaming = config.streaming
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.reset_stats()
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'total_tokens_generated': 0,
            'total_time': 0.0,
            'tokens_per_second': 0.0,
            'avg_latency_per_token': 0.0,
            'memory_usage': 0.0
        }
    
    def warmup(self, input_ids: torch.Tensor):
        """Warmup the model for consistent performance"""
        self.logger.info("Warming up model...")
        
        # Run a few forward passes to warm up
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(input_ids=input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.logger.info("Model warmup completed")
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_full_text: bool = True
    ) -> Union[torch.Tensor, List[str]]:
        """Generate text from input"""
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if do_sample is None:
            do_sample = self.do_sample
        if temperature is None:
            temperature = self.temperature
        if top_k is None:
            top_k = self.top_k
        if top_p is None:
            top_p = self.top_p
        if repetition_penalty is None:
            repetition_penalty = self.repetition_penalty
        
        batch_size, seq_len = input_ids.shape
        
        # Move to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Initialize KV cache
        if self.use_kv_cache:
            past_key_values = tuple([None] * len(self.model.layers))
        else:
            past_key_values = None
        
        # Generation loop
        generated_tokens = []
        current_input = input_ids
        
        start_time = time.time()
        
        for i in range(max_new_tokens):
            # Forward pass
            outputs = self.model(
                input_ids=current_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=self.use_kv_cache,
                output_attentions=False
            )
            
            logits = outputs['last_hidden_state'][:, -1, :]  # Get last token logits
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, current_input, repetition_penalty)
            
            # Sampling
            if do_sample:
                next_token = self._sample_tokens(
                    logits, temperature, top_k, top_p
                )
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_tokens.append(next_token)
            current_input = next_token
            
            # Update attention mask if needed
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones(batch_size, 1, device=self.device)
                ], dim=1)
            
            # Update KV cache
            if self.use_kv_cache:
                past_key_values = outputs.get('past_key_values', past_key_values)
        
        end_time = time.time()
        
        # Combine input and generated tokens
        if return_full_text:
            all_tokens = torch.cat([input_ids] + generated_tokens, dim=1)
        else:
            all_tokens = torch.cat(generated_tokens, dim=1)
        
        # Update statistics
        total_tokens = all_tokens.numel()
        total_time = end_time - start_time
        self.stats['total_tokens_generated'] += total_tokens
        self.stats['total_time'] += total_time
        self.stats['tokens_per_second'] = self.stats['total_tokens_generated'] / max(self.stats['total_time'], 1e-6)
        self.stats['avg_latency_per_token'] = self.stats['total_time'] / max(self.stats['total_tokens_generated'], 1e-6)
        
        if torch.cuda.is_available():
            self.stats['memory_usage'] = torch.cuda.memory_allocated() / 1024**3  # GB
        
        if self.tokenizer:
            return self.tokenizer.batch_decode(all_tokens, skip_special_tokens=True)
        else:
            return all_tokens
    
    @torch.no_grad()
    def generate_streaming(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[Union[str, torch.Tensor], None, None]:
        """Generate text token by token for streaming"""
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        batch_size, seq_len = input_ids.shape
        input_ids = input_ids.to(self.device)
        
        # Initialize KV cache
        if self.use_kv_cache:
            past_key_values = tuple([None] * len(self.model.layers))
        else:
            past_key_values = None
        
        current_input = input_ids
        generated_tokens = []
        
        for i in range(max_new_tokens):
            outputs = self.model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=self.use_kv_cache,
                output_attentions=False
            )
            
            logits = outputs['last_hidden_state'][:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_tokens.append(next_token)
            current_input = next_token
            
            if self.use_kv_cache:
                past_key_values = outputs.get('past_key_values', past_key_values)
            
            # Yield token or text
            if self.tokenizer:
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                yield token_text
            else:
                yield next_token
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        for i in range(logits.shape[0]):
            for token_id in set(input_ids[i].tolist()):
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty
        
        return logits
    
    def _sample_tokens(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """Sample tokens using temperature, top-k, and top-p sampling"""
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, top_k)
            min_top_k = top_k_logits[:, -1:].expand_as(logits)
            logits = torch.where(logits < min_top_k, torch.full_like(logits, float('-inf')), logits)
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Shift right to keep at least one token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def get_generation_params(self) -> Dict[str, Union[int, float, bool]]:
        """Get current generation parameters"""
        return {
            'max_new_tokens': self.max_new_tokens,
            'do_sample': self.do_sample,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'repetition_penalty': self.repetition_penalty,
            'use_kv_cache': self.use_kv_cache,
            'use_fp8_inference': self.use_fp8_inference,
            'batch_size': self.batch_size
        }
    
    def update_generation_params(self, **kwargs):
        """Update generation parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """Get performance statistics"""
        return self.stats.copy()
    
    def save_stats(self, filepath: str):
        """Save performance statistics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def load_model_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded model from {checkpoint_path}")
    
    def optimize_for_inference(self):
        """Apply optimizations for inference"""
        self.model.eval()
        
        # Enable torch.jit optimization if available
        if hasattr(torch.jit, 'optimize_for_inference'):
            try:
                self.model = torch.jit.optimize_for_inference(self.model)
                self.logger.info("Applied JIT optimization for inference")
            except Exception as e:
                self.logger.warning(f"JIT optimization failed: {e}")
        
        # Enable memory efficient attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            self.logger.info("Memory efficient attention is available")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BatchInferenceEngine(InferenceEngine):
    """Batch inference engine for processing multiple inputs efficiently"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @torch.no_grad()
    def generate_batch(
        self,
        input_ids_list: List[torch.Tensor],
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> List[Union[str, torch.Tensor]]:
        """Generate text for multiple inputs in batch"""
        
        # Pad inputs to same length
        max_len = max(tensor.size(1) for tensor in input_ids_list)
        
        padded_inputs = []
        for tensor in input_ids_list:
            if tensor.size(1) < max_len:
                pad_size = max_len - tensor.size(1)
                padding = torch.zeros(tensor.size(0), pad_size, dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, padding], dim=1)
            padded_inputs.append(tensor)
        
        # Stack into batch
        batch_input = torch.cat(padded_inputs, dim=0)
        
        # Generate
        results = self.generate(batch_input, max_new_tokens, **kwargs)
        
        # Split results back
        batch_size = batch_input.size(0)
        result_size = results.size(0) // batch_size
        
        if self.tokenizer:
            # Split text results
            return [results[i*result_size:(i+1)*result_size] for i in range(batch_size)]
        else:
            # Split tensor results
            return [results[i*result_size:(i+1)*result_size] for i in range(batch_size)]


def create_inference_engine(
    model: Union[DeepSeekModel, DeepSeekMoEModel],
    config: InferenceConfig,
    tokenizer=None
) -> InferenceEngine:
    """Create inference engine with optimal settings"""
    
    if config.batch_size > 1:
        return BatchInferenceEngine(model, config, tokenizer)
    else:
        return InferenceEngine(model, config, tokenizer)