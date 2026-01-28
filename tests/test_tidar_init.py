
import torch
from better_ai.config import ModelConfig
from better_ai.models.enhanced_model import EnhancedDeepSeekModel

def test_tidar_initialization():
    config = ModelConfig(
        use_tidar=True,
        tidar_num_steps=3,
        tidar_diffusion_dim=64,
        hidden_dim=128,
        vocab_size=100,
        num_layers=1
    )
    model = EnhancedDeepSeekModel(config)

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    outputs = model(input_ids, return_advanced_features=True)

    assert "tidar" in outputs["advanced_features"]
    tidar_out = outputs["advanced_features"]["tidar"]
    assert "refined_scratchpad" in tidar_out
    assert tidar_out["refined_scratchpad"].shape == (batch_size, seq_len, config.hidden_dim)
    print("TiDAR initialization and forward pass successful!")

if __name__ == "__main__":
    test_tidar_initialization()
