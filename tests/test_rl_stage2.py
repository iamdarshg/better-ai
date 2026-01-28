
import torch
from better_ai.config import ModelConfig, TrainingConfig
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.training.enhanced_trainer import EnhancedMoETrainer

def test_rl_stage2_forward():
    config = ModelConfig(
        vocab_size=100,
        hidden_dim=128,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2
    )
    train_config = TrainingConfig(rl_stage=2)
    model = EnhancedDeepSeekModel(config)

    trainer = EnhancedMoETrainer(
        model=model,
        train_dataloader=None,
        eval_dataloader=None,
        optimizer=None,
        scheduler=None,
        config=train_config,
        device=torch.device("cpu"),
        use_enhanced_features=False
    )

    batch = {
        "prompt": "Test prompt",
        "response": "Test response",
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones((2, 8))
    }

    loss, aux_loss, expert_ids = trainer._enhanced_forward_pass(batch)
    print(f"RL Stage 2 forward pass successful! Loss: {loss.item()}")
    assert loss.item() != 0

if __name__ == "__main__":
    test_rl_stage2_forward()
