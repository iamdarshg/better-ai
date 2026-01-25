#!/usr/bin/env python
"""Test EnhancedDeepSeekModel forward pass"""

import torch
from better_ai.config import ModelConfig
from better_ai.models.enhanced_model import EnhancedDeepSeekModel

config = ModelConfig()
print(f'Testing EnhancedDeepSeekModel with:')
print(f'  hidden_dim={config.hidden_dim}')
print(f'  num_attention_heads={config.num_attention_heads}')
print(f'  num_key_value_heads={config.num_key_value_heads}')

# Create enhanced model
model = EnhancedDeepSeekModel(
    config=config,
    device='cuda',
)
print(f'Model created successfully')

# Test forward pass
try:
    input_ids = torch.randint(0, config.vocab_size, (2, 512)).cuda()
    with torch.no_grad():
        output = model.forward(input_ids=input_ids)
    print(f'Success! Output keys: {output.keys()}')
    print(f'Logits shape: {output["logits"].shape}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
