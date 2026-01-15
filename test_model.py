#!/usr/bin/env python
"""Test model forward pass to diagnose dimension mismatch"""

import torch
from better_ai.config import ModelConfig
from better_ai.models.core import DeepSeekModel

config = ModelConfig()
print(f'Testing DeepSeekModel with:')
print(f'  hidden_dim={config.hidden_dim}')
print(f'  num_attention_heads={config.num_attention_heads}')
print(f'  num_key_value_heads={config.num_key_value_heads}')
print(f'  head_dim={config.hidden_dim // config.num_attention_heads}')

# Create core model only
model = DeepSeekModel(
    vocab_size=config.vocab_size,
    hidden_size=config.hidden_dim,
    num_layers=2,  # Just 2 layers for testing
    num_heads=config.num_attention_heads,
    num_key_value_heads=config.num_key_value_heads,
    intermediate_size=config.intermediate_dim,
    max_seq_length=config.max_seq_length,
    dropout=config.residual_dropout,
)
model.cuda()

# Test forward pass
try:
    input_ids = torch.randint(0, config.vocab_size, (2, 512)).cuda()
    with torch.no_grad():
        output = model(input_ids=input_ids)
    print(f'Output type: {type(output)}')
    print(f'Output: {output}')
    if isinstance(output, dict):
        print(f'Output keys: {output.keys()}')
        print(f'Logits shape: {output.get("logits", output.get("last_hidden_state")).shape}')
    else:
        print(f'Success! Output shape: {output[0].shape}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
