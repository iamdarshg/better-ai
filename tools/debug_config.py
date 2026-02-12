import sys

sys.path.insert(0, ".")
sys.path.insert(0, "./tools")
from update_readme_estimates import calculate_all_parameters
from better_ai.config import ModelConfig

# Check default vs production config
print("DEFAULT CONFIG:")
default_config = ModelConfig()
print(f"  vocab_size: {default_config.vocab_size}")
print(f"  hidden_dim: {default_config.hidden_dim}")
print(f"  num_layers: {default_config.num_layers}")
print(f"  intermediate_dim: {default_config.intermediate_dim}")
print(f"  num_experts: {default_config.num_experts}")
print(f"  use_tidar: {default_config.use_tidar}")
print(f"  tidar_num_layers: {default_config.tidar_num_layers}")
print(f"  tidar_diffusion_dim: {default_config.tidar_diffusion_dim}")

default_params = calculate_all_parameters(default_config)
print(f"  Total params: {default_params['total_params'] / 1e9:.2f}B")

print("\nPRODUCTION CONFIG:")
prod_config = ModelConfig.get_production_config()
print(f"  vocab_size: {prod_config.vocab_size}")
print(f"  hidden_dim: {prod_config.hidden_dim}")
print(f"  num_layers: {prod_config.num_layers}")
print(f"  intermediate_dim: {prod_config.intermediate_dim}")
print(f"  num_experts: {prod_config.num_experts}")
print(f"  use_tidar: {prod_config.use_tidar}")
print(f"  tidar_num_layers: {prod_config.tidar_num_layers}")
print(f"  tidar_diffusion_dim: {prod_config.tidar_diffusion_dim}")

prod_params = calculate_all_parameters(prod_config)
print(f"  Total params: {prod_params['total_params'] / 1e9:.2f}B")

print("\n" + "=" * 60)
print("BREAKDOWN COMPARISON:")
print("=" * 60)
print(f"{'Component':<30} {'Default':<15} {'Production':<15}")
print("-" * 60)
for key in [
    "embedding_params",
    "attention_total",
    "ffn_standard_total",
    "ffn_moe_all_total",
    "total_feature_params",
]:
    d_val = default_params[key]
    p_val = prod_params[key]
    print(f"{key:<30} {d_val / 1e9:>10.2f}B    {p_val / 1e9:>10.2f}B")
