import sys

sys.path.insert(0, ".")
sys.path.insert(0, "./tools")
from update_readme_estimates import (
    calculate_inference_vram,
    calculate_training_vram,
    calculate_all_parameters,
    find_max_batch_size,
    GPU_SPECS,
)
from better_ai.config import ModelConfig, TrainingConfig, InferenceConfig

config = ModelConfig.get_production_config()
train_config = TrainingConfig()
inf_config = InferenceConfig()
params = calculate_all_parameters(config)

print("=" * 60)
print("INFERENCE VRAM ANALYSIS")
print("=" * 60)

# Check a specific batch size for RTX 2070
for batch in [1, 38, 100]:
    vram = calculate_inference_vram(config, inf_config, "bf16", batch)
    print(f"\nBatch size {batch}:")
    print(f"  Total: {vram['total_gb']:.2f} GB")
    print(f"  Model: {vram['model_gb']:.2f} GB")
    print(f"  KV Cache: {vram['kv_cache_gb']:.2f} GB")
    print(f"  Activations: {vram['activations_gb']:.2f} GB")
    print(f"  Overhead: {vram['overhead_gb']:.2f} GB")
    print(f"  Fragmentation: {vram['fragmentation_gb']:.2f} GB")

print("\n" + "=" * 60)
print("TRAINING VRAM ANALYSIS")
print("=" * 60)

# Check training VRAM for different batch sizes
for batch in [1, 10, 100, 260]:
    vram = calculate_training_vram(config, train_config, "bf16", batch)
    print(f"\nBatch size {batch}:")
    print(f"  Total: {vram['total_gb']:.2f} GB")
    print(f"  Weights: {vram['model_weights_gb']:.2f} GB")
    print(f"  Master Weights: {vram['master_weights_gb']:.2f} GB")
    print(f"  Gradients: {vram['gradients_gb']:.2f} GB")
    print(f"  Optimizer: {vram['optimizer_states_gb']:.2f} GB")
    print(f"  Activations: {vram['activations_gb']:.2f} GB")
    print(f"  Comm Buffers: {vram['comm_buffers_gb']:.2f} GB")

print("\n" + "=" * 60)
print("MAX BATCH SIZE VERIFICATION")
print("=" * 60)

for gpu_name in ["RTX 2070", "H300e"]:
    gpu = GPU_SPECS[gpu_name]
    print(f"\n{gpu_name} ({gpu.vram_gb}GB):")

    # Test inference max batch
    max_inf_bf16 = find_max_batch_size(
        config, train_config, inf_config, gpu_name, "bf16", "inference"
    )
    vram_inf = calculate_inference_vram(config, inf_config, "bf16", max_inf_bf16)
    print(f"  Inference BF16 max batch: {max_inf_bf16} -> {vram_inf['total_gb']:.1f}GB")

    # Test training max batch
    max_train_bf16 = find_max_batch_size(
        config, train_config, inf_config, gpu_name, "bf16", "training"
    )
    if max_train_bf16 > 0:
        vram_train = calculate_training_vram(
            config, train_config, "bf16", max_train_bf16
        )
        print(
            f"  Training BF16 max batch: {max_train_bf16} -> {vram_train['total_gb']:.1f}GB"
        )
    else:
        print(f"  Training BF16: Doesn't fit!")

print("\n" + "=" * 60)
print("CONFIG VALUES")
print("=" * 60)
print(f"Model params: {params['total_params'] / 1e9:.2f}B")
print(f"Inference max_seq_length: {inf_config.max_seq_length}")
print(f"Training max_seq_length: {train_config.max_seq_length}")
print(f"Training batch_size: {train_config.batch_size}")
print(f"Gradient accumulation: {train_config.gradient_accumulation_steps}")
