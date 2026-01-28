"""
Testing configuration and fixtures
Provides reduced-size configurations for faster testing with 32-64x parameter reductions
"""

import pytest
from better_ai.config import ModelConfig, TrainingConfig, InferenceConfig


@pytest.fixture
def small_model_config():
    """
    Reduced ModelConfig for testing - 32-64x smaller parameters.
    Suitable for component verification without significant compute.
    """
    return ModelConfig(
        # Architecture parameters - reduced 32-64x
        vocab_size=2048,  # 64000 -> 2048
        hidden_dim=256,   # 8192 -> 256
        num_layers=2,     # 32 -> 2
        num_attention_heads=8,  # 64 -> 8
        num_key_value_heads=4,  # 32 -> 4
        intermediate_dim=24,    # 768 -> 24 (4x hidden_dim)
        max_seq_length=2048,    # 262144 -> 2048
        
        # MoE parameters - reduced
        num_experts=4,  # 18 -> 4
        num_experts_per_token=2,  # Keep same
        expert_capacity_factor=1.25,  # Keep same
        shared_experts=1,  # Keep same
        moe_load_balance_weight=0.01,  # Keep same
        
        # Attention parameters
        rope_theta=10000.0,  # Keep same
        rope_dim=None,  # Keep same
        attention_dropout=0.0,  # Keep same
        residual_dropout=0.0,  # Keep same
        embedding_dropout=0.0,  # Keep same
        
        # Normalization
        norm_type="rmsnorm",  # Keep same
        norm_eps=1e-6,  # Keep same
        
        # Activation
        activation="swiglu",  # Keep same
        
        # Initialization
        init_std=0.02,  # Keep same
        init_method="normal",  # Keep same
        
        # Quantization
        use_fp8=True,  # Keep same
        fp8_e4m3=True,  # Keep same
        
        # Sparse attention
        use_sparse_attention=False,  # Disable for faster testing
        local_window_size=256,  # 8192 -> 256
        global_stride=32,  # 1024 -> 32
        
        # Memory optimization
        use_gradient_checkpointing=False,  # Disable for testing
        use_flash_attention=True,  # Keep same
        use_paged_attention=False,  # Disable for testing
        
        # Ring Attention parameters
        use_ring_attention=False,  # Disable for testing
        ring_block_size=256,  # 1024 -> 256
        ring_num_devices=1,  # Single device for testing
        
        # Linear Attention parameters
        use_linear_attention=False,  # Keep same
        
        # CoT Specialization parameters
        use_cot_specialization=False,  # Disable for faster testing
        cot_num_heads=8,  # 64 -> 8
        cot_hidden_dim=24,  # 768 -> 24
        
        # Inner Monologue parameters
        use_inner_monologue=False,  # Disable for faster testing
        thought_token_id=None,  # Keep same
        private_subspace_dim=128,  # 4096 -> 128
        
        # STaR parameters
        use_star=False,  # Disable for faster testing
        star_bootstrap_rounds=2,  # 3 -> 2
        star_consistency_samples=2,  # 10 -> 2
        
        # Tool-Use parameters
        use_tool_heads=False,  # Disable for faster testing
        tool_vocab_size=100,  # 1000 -> 100
        tool_hidden_dim=24,  # 192 -> 24
        
        # JSON+DBOps Head parameters
        use_json_db_ops_head=False,  # Disable for testing
        json_db_ops_ratio=0.1,  # Keep same
        json_db_ops_internal_dim=32,  # 256 -> 32
        
        # Math Reasoning Head parameters
        use_math_reasoning_head=False,  # Disable for testing
        math_reasoning_ratio=0.1,  # Keep same
        math_reasoning_internal_dim=32,  # 256 -> 32
        
        # Algorithm Head parameters
        use_algorithm_head=False,  # Disable for testing
        algorithm_ratio=0.1,  # Keep same
        algorithm_internal_dim=32,  # 256 -> 32
        
        # Grammar Constraint parameters
        use_grammar_constraints=False,  # Disable for testing
        grammar_type="gbnf",  # Keep same
        enforce_json_output=False,  # Disable for testing
        
        # Entropic Steering parameters
        use_entropic_steering=False,  # Disable for testing
        entropy_threshold=2.5,  # Keep same
        clarify_token_id=None,  # Keep same
        
        # Recursive Scratchpad parameters
        use_recursive_scratchpad=False,  # Disable for testing
        scratchpad_max_iterations=2,  # 8 -> 2
        scratchpad_hidden_dim=256,  # 8192 -> 256
    )


@pytest.fixture
def small_training_config():
    """
    Reduced TrainingConfig for testing.
    Suitable for verifying training loops with minimal compute.
    """
    return TrainingConfig(
        # Basic training - reduced for testing
        batch_size=1,  # Keep minimal
        gradient_accumulation_steps=1,  # 4 -> 1
        learning_rate=1e-4,  # Keep same
        warmup_steps=1,  # 1 -> 1
        max_steps=10,  # 1000000 -> 10
        save_steps=5,  # 10 -> 5
        eval_steps=5,  # 1000 -> 5
        
        # Optimizer
        optimizer="adamw",  # Keep same
        beta1=0.9,  # Keep same
        beta2=0.95,  # Keep same
        weight_decay=0.1,  # Keep same
        eps=1e-8,  # Keep same
        use_8bit_optimizer=False,  # Disable for testing
        
        # LR scheduling
        lr_schedule="cosine",  # Keep same
        lr_decay_steps=None,  # Keep same
        min_lr_ratio=0.1,  # Keep same
        
        # FP8 specific
        fp8_loss_scale=1.0,  # Keep same
        fp8_delayed_scaling=False,  # Disable for testing
        fp8_scaling_window=16,  # Keep same
        
        # Data - reduced
        data_path="./data",  # Keep same
        tokenizer_path=None,  # Keep same
        max_seq_length=512,  # 8192 -> 512
        shuffle_buffer_size=100,  # 10000 -> 100
        
        # Logging
        log_dir="./logs",  # Keep same
        log_every_n_steps=1,  # 100 -> 1
        wandb_project=None,  # Disable for testing
        wandb_entity=None,  # Keep same
        
        # Checkpointing
        output_dir="./checkpoints",  # Keep same
        save_total_limit=2,  # 5 -> 2
        save_strategy="steps",  # Keep same
        
        # Mixed precision
        fp16=False,  # Keep same
        bf16=False,  # Disable for testing
        
        # Distributed training
        distributed_backend="ddp",  # Keep same
        fsdp_sharding_strategy="FULL_SHARD",  # Keep same
        fsdp_cpu_offload=False,  # Disable for testing
        
        # Monitoring
        profile_memory=False,  # Disable for testing
        profile_time=False,  # Disable for testing
        
        # Pruning
        pruning_ratio=0.1,  # Keep same
        pruning_steps=None,  # Keep same
        
        # Ring Attention
        use_ring_attention=False,  # Disable for testing
    )


@pytest.fixture
def small_inference_config():
    """
    Reduced InferenceConfig for testing.
    """
    return InferenceConfig(
        # Generation
        max_new_tokens=64,  # 512 -> 64
        do_sample=False,  # Disable for deterministic testing
        temperature=0.8,  # Keep same
        top_k=10,  # 50 -> 10
        top_p=0.9,  # Keep same
        repetition_penalty=1.2,  # Keep same
        
        # Optimization
        use_kv_cache=False,  # Disable for testing
        use_fp8_inference=False,  # Disable for testing
        batch_size=1,  # Keep same
        streaming=False,  # Keep same
        
        # Memory
        max_batch_size=2,  # 32 -> 2
        max_seq_length=512,  # 8192 -> 512
        cache_size=None,  # Keep same
        
        # Quantization
        quantize_weights=False,  # Keep same
        quantize_activations=False,  # Keep same
        weight_bits=8,  # Keep same
        activation_bits=8,  # Keep same
        
        # Serving
        serve_port=8080,  # Keep same
        serve_host="0.0.0.0",  # Keep same
        max_concurrent_requests=1,  # 10 -> 1
    )


@pytest.fixture
def model_config():
    """Default model config for tests that don't need reduced sizes."""
    return ModelConfig()


@pytest.fixture
def training_config():
    """Default training config for tests that don't need reduced sizes."""
    return TrainingConfig()


@pytest.fixture
def inference_config():
    """Default inference config for tests that don't need reduced sizes."""
    return InferenceConfig()
