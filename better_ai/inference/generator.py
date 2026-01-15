"""Text generation utilities for DeepSeek model"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 512
    min_new_tokens: int = 1
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0
    pad_token_id: int = 0
    eos_token_id: int = 1
    use_cache: bool = True


class TextGenerator:
    """Text generator for DeepSeek models"""
    
    def __init__(self, model, tokenizer, config: Optional[GenerationConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        
    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate text given input IDs"""
        
        config = generation_config or self.config
        batch_size = input_ids.shape[0]
        
        # Initialize
        generated_ids = input_ids.clone()
        past_key_values = None
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        for step in range(config.max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    generated_ids,
                    past_key_values=past_key_values,
                    use_cache=config.use_cache
                )
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
                
                # Get next token logits
                next_token_logits = logits[:, -1, :]
                
                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    for i in range(batch_size):
                        if not finished_sequences[i]:
                            for token_id in set(generated_ids[i].tolist()):
                                next_token_logits[i, token_id] /= config.repetition_penalty
                
                # Apply temperature
                if config.temperature > 0:
                    next_token_logits = next_token_logits / config.temperature
                
                # Apply top-k and top-p filtering
                if config.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, config.top_k, dim=-1)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
                
                if config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample or take argmax
                if config.do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Update generated sequences
                next_tokens = next_tokens.unsqueeze(-1)
                generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
                
                # Check for EOS tokens
                finished_sequences = finished_sequences | (next_tokens.squeeze(-1) == config.eos_token_id)
                
                # Stop if all sequences are finished
                if finished_sequences.all():
                    break
        
        return generated_ids
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs to text"""
        texts = []
        for i in range(token_ids.shape[0]):
            text = self.tokenizer.decode(token_ids[i].tolist())
            texts.append(text)
        return texts
    
    def __call__(self, text: str, **kwargs) -> str:
        """Generate text from input string"""
        # Encode input
        input_ids = torch.tensor(
            [self.tokenizer.encode(text)], 
            dtype=torch.long,
            device=next(self.model.parameters()).device
        )
        
        # Generate
        with torch.no_grad():
            generated_ids = self.generate(input_ids, **kwargs)
        
        # Decode
        generated_text = self.decode(generated_ids)[0]
        
        # Remove input text from generated text
        if generated_text.startswith(text):
            generated_text = generated_text[len(text):]
        
        return generated_text


def create_text_generator(model, tokenizer, **config_kwargs) -> TextGenerator:
    """Create a text generator with default configuration"""
    config = GenerationConfig(**config_kwargs)
    return TextGenerator(model, tokenizer, config)