"""
Model configuration for DeepSeek V3.2 inspired toy model
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class ModelConfig:
    """Configuration for the transformer model"""
    
    # Architecture parameters
    vocab_size: int = 64000  # Increased for better coding coverage
    hidden_dim: int = 1024  # Increased for better representation
    num_layers: int = 12  # Reduced for efficiency
    num_attention_heads: int = 32  # Increased proportionally
    num_key_value_heads: Optional[int] = 16  # For GQA, maintain 2:1 ratio
    intermediate_dim: int = 768  # 4x hidden_dim for SwiGLU
    max_seq_length: int = 2048  # Increased with Ring Attention
    
    # MoE parameters
    num_experts: int = 16  # Reduced for efficiency
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    shared_experts: int = 1
    moe_load_balance_weight: float = 0.01
    
    # Attention parameters
    rope_theta: float = 10000.0
    rope_dim: Optional[int] = None  # Default: hidden_dim // num_attention_heads
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    embedding_dropout: float = 0.0
    
    # Normalization
    norm_type: str = "rmsnorm"  # "rmsnorm" or "layernorm"
    norm_eps: float = 1e-6
    
    # Activation
    activation: str = "swiglu"  # "swiglu", "gelu", "relu"
    
    # Initialization
    init_std: float = 0.02
    init_method: str = "normal"
    
    # Quantization
    use_fp8: bool = True
    fp8_e4m3: bool = True  # E4M3 for forward, E5M2 for gradients
    
    # Sparse attention
    use_sparse_attention: bool = True
    local_window_size: int = 1024
    global_stride: int = 512
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    use_paged_attention: bool = True
    
    # Ring Attention parameters
    use_ring_attention: bool = True
    ring_block_size: int = 2048
    ring_num_devices: Optional[int] = None  # Auto-detect
    
    # Linear Attention parameters
    use_linear_attention: bool = False

    # CoT Specialization parameters
    use_cot_specialization: bool = True
    cot_num_heads: int = 64
    cot_hidden_dim: int = 768
    
    # Inner Monologue parameters
    use_inner_monologue: bool = True
    thought_token_id: Optional[int] = None  # Will be set during tokenization
    private_subspace_dim: int = 4096
    
    # STaR parameters
    use_star: bool = True
    star_bootstrap_rounds: int = 3
    star_consistency_samples: int = 10
    
    # Tool-Use parameters
    use_tool_heads: bool = True
    tool_vocab_size: int = 1000  # Number of tool tokens
    tool_hidden_dim: int = 192

    # JSON+DBOps Head parameters
    use_json_db_ops_head: bool = False
    json_db_ops_ratio: float = 0.1
    json_db_ops_internal_dim: int = 256

    # Math Reasoning Head parameters
    use_math_reasoning_head: bool = False
    math_reasoning_ratio: float = 0.1
    math_reasoning_internal_dim: int = 256

    # Algorithm Head parameters
    use_algorithm_head: bool = False
    algorithm_ratio: float = 0.1
    algorithm_internal_dim: int = 256
    
    # Grammar Constraint parameters
    use_grammar_constraints: bool = True
    grammar_type: str = "gbnf"  # "gbnf" or "none"
    enforce_json_output: bool = True
    
    # Entropic Steering parameters
    use_entropic_steering: bool = True
    entropy_threshold: float = 2.5
    clarify_token_id: Optional[int] = None  # Will be set during tokenization
    
    # Recursive Scratchpad parameters
    use_recursive_scratchpad: bool = True
    scratchpad_max_iterations: int = 8
    scratchpad_hidden_dim: int = 8192

@dataclass
class TrainingConfig:
    """Configuration for training"""
    
    # Basic training
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 1
    max_steps: int = 100000
    save_steps: int = 10
    eval_steps: int = 1000
    
    # Optimizer
    optimizer: str = "adamw"  # "adamw", "lion", "adafactor"
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    eps: float = 1e-8
    use_8bit_optimizer: bool = True
    
    # LR scheduling
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant"
    lr_decay_steps: Optional[int] = None
    min_lr_ratio: float = 0.1
    
    # FP8 specific
    fp8_loss_scale: float = 1.0
    fp8_delayed_scaling: bool = True
    fp8_scaling_window: int = 16
    
    # Data
    data_path: str = "./data"
    tokenizer_path: Optional[str] = None
    max_seq_length: int = 8192
    shuffle_buffer_size: int = 10000
    
    # Logging
    log_dir: str = "./logs"
    log_every_n_steps: int = 100
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_total_limit: int = 5
    save_strategy: str = "steps"  # "steps" or "epoch"
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Distributed training
    distributed_backend: str = "ddp"  # "ddp", "fsdp"
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = True
    
    # Monitoring
    profile_memory: bool = True
    profile_time: bool = True

    # Pruning
    pruning_ratio: float = 0.1
    pruning_steps: Optional[List[int]] = None

    # Ring Attention
    use_ring_attention: bool = False

@dataclass
class InferenceConfig:
    """Configuration for inference"""
    
    # Generation
    max_new_tokens: int = 512
    do_sample: bool = True
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    
    # Optimization
    use_kv_cache: bool = True
    use_fp8_inference: bool = True
    batch_size: int = 1
    streaming: bool = False
    
    # Memory
    max_batch_size: int = 32
    max_seq_length: int = 8192
    cache_size: Optional[int] = None
    
    # Quantization
    quantize_weights: bool = False
    quantize_activations: bool = False
    weight_bits: int = 8
    activation_bits: int = 8
    
    # Serving
    serve_port: int = 8080
    serve_host: str = "0.0.0.0"
    max_concurrent_requests: int = 10
