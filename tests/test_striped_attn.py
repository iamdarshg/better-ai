
import torch
from better_ai.config import ModelConfig
from better_ai.models.enhanced_model import EnhancedDeepSeekModel

def test_striped_attention_init():
    config = ModelConfig(
        use_ring_attention=True,
        use_striped_attention=True,
        hidden_dim=128,
        num_layers=1,
        vocab_size=100
    )
    model = EnhancedDeepSeekModel(config)

    # Check if attention is StripedAttention
    from better_ai.models.ring_attention import StripedAttention
    assert isinstance(model.model.layers[0].self_attn, StripedAttention)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    # Forward pass
    outputs = model(input_ids)
    assert outputs["logits"].shape == (batch_size, seq_len, 100)
    print("Striped Attention initialization and forward pass successful!")

if __name__ == "__main__":
    test_striped_attention_init()
