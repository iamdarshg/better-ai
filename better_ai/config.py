"""
Model configuration for DeepSeek V3.2 inspired toy model
"""

import math
import json
import yaml
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Union, Dict, Any
from .utils.exceptions import ConfigError


@dataclass
class ModelConfig:
    """Configuration for the transformer model"""

    vocab_size: int = 64000
    hidden_dim: int = 4096
    num_layers: int = 8
    num_attention_heads: int = 24
    num_key_value_heads: Optional[int] = 4  # Default: num_attention_heads // 2
    intermediate_dim: int = 11000
    max_seq_length: int = 524288

    # MoE parameters
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.1
    shared_experts: int = 1
    moe_load_balance_weight: float = 0.01
    use_moe_every_n_layers: int = 2

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
    use_sparse_attention: bool = False
    local_window_size: int = 4096
    global_stride: int = 512

    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    use_paged_attention: bool = False

    # Ring Attention parameters
    use_ring_attention: bool = False
    use_striped_attention: bool = True
    ring_block_size: int = 1024
    ring_num_devices: Optional[int] = None  # Auto-detect

    # Linear Attention parameters
    use_linear_attention: bool = False

    # CoT Specialization parameters
    use_cot_specialization: bool = False
    cot_num_heads: int = 4
    cot_hidden_dim: int = 3072

    # Inner Monologue parameters
    use_inner_monologue: bool = False
    thought_token_id: Optional[int] = 100  # Default for testing
    thought_end_token_id: Optional[int] = 101  # Default for testing
    private_subspace_dim: int = 3072

    # STaR parameters
    use_star: bool = True
    star_bootstrap_rounds: int = 3
    star_consistency_samples: int = 8

    # Tool-Use parameters
    use_tool_heads: bool = True
    tool_vocab_size: int = 6144  # Number of tool tokens
    tool_hidden_dim: int = 2048

    # JSON+DBOps Head parameters
    use_json_db_ops_head: bool = True
    json_db_ops_ratio: float = 0.125
    json_db_ops_internal_dim: int = 1024

    # Math Reasoning Head parameters
    use_math_reasoning_head: bool = True
    math_reasoning_ratio: float = 0.11
    math_reasoning_internal_dim: int = 2048

    # Algorithm Head parameters
    use_algorithm_head: bool = True
    algorithm_ratio: float = 0.1
    algorithm_internal_dim: int = 2048

    # Grammar Constraint parameters
    use_grammar_constraints: bool = True
    grammar_type: str = "gbnf"  # "gbnf" or "none"
    enforce_json_output: bool = False

    # Entropic Steering parameters
    use_entropic_steering: bool = True
    entropy_threshold: float = 2.5
    clarify_token_id: Optional[int] = None  # Will be set during tokenization

    # Recursive Scratchpad parameters
    use_recursive_scratchpad: bool = True
    scratchpad_max_iterations: int = 6
    scratchpad_hidden_dim: int = 4096

    # TiDAR parameters
    use_tidar: bool = True
    tidar_num_steps: int = 2
    tidar_diffusion_dim: int = 1536
    tidar_num_layers: int = 2

    # Feature Toggles for Memory management
    use_reward_models: bool = True
    use_reasoning_rewards: bool = True
    use_value_head: bool = True

    # Resource Estimation Ratios
    training_fragmentation_ratio: float = 1.10
    inference_fragmentation_ratio: float = 1.15

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate configuration parameters"""
        if self.vocab_size <= 0:
            raise ConfigError("vocab_size must be positive")
        if self.hidden_dim <= 0:
            raise ConfigError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ConfigError("num_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ConfigError("num_attention_heads must be positive")
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ConfigError("hidden_dim must be divisible by num_attention_heads")

        if self.num_experts < 0:
            raise ConfigError("num_experts cannot be negative")

        if self.num_key_value_heads is not None:
            if self.num_key_value_heads > self.num_attention_heads:
                raise ConfigError(
                    "num_key_value_heads cannot be greater than num_attention_heads"
                )

        if self.num_experts_per_token > self.num_experts + self.shared_experts:
            raise ConfigError(
                "num_experts_per_token cannot be greater than total experts"
            )

        if self.use_ring_attention and self.ring_block_size <= 0:
            raise ConfigError(
                "ring_block_size must be positive if ring attention is enabled"
            )

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate HTSR configuration"""
        if self.htsr_monitor_interval < 1:
            raise ConfigError("htsr_monitor_interval must be positive")
        if self.htsr_alpha_upper_threshold < 1.0:
            raise ConfigError("htsr_alpha_upper_threshold must be >= 1.0")
        if self.htsr_variance_threshold < 0:
            raise ConfigError("htsr_variance_threshold must be non-negative")
        if not 0 < self.htsr_lr_reduction_factor <= 1.0:
            raise ConfigError("htsr_lr_reduction_factor must be between 0 and 1")
        if self.htsr_wd_increase_factor < 1.0:
            raise ConfigError("htsr_wd_increase_factor must be >= 1.0")
        if self.htsr_dashboard_port < 1 or self.htsr_dashboard_port > 65535:
            raise ConfigError("htsr_dashboard_port must be between 1 and 65535")
        if self.htsr_dashboard_auto_refresh < 10:
            raise ConfigError("htsr_dashboard_auto_refresh must be >= 10 seconds")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def get_production_config(cls):
        """Returns a production-ready configuration with larger dimensions"""
        return cls()

    @classmethod
    def get_small_model_config(cls):
        """Returns a minimal configuration for CI/Testing safety with all features initialized"""
        return cls(
            vocab_size=1024,
            hidden_dim=48,
            num_layers=2,
            num_attention_heads=6,
            num_key_value_heads=2,
            intermediate_dim=24,
            max_seq_length=128,
            num_experts=4,
            num_experts_per_token=2,
            shared_experts=1,
            use_ring_attention=False,
            use_striped_attention=False,
            use_tidar=False,
            use_star=True,
            use_recursive_scratchpad=True,
            scratchpad_max_iterations=2,
            use_grammar_constraints=False,
            use_cot_specialization=True,
            cot_num_heads=2,
            cot_hidden_dim=8,
            use_tool_heads=True,
            tool_vocab_size=10,
            tool_hidden_dim=24,
        )

    def to_file(self, filepath: str):
        """Save config to file"""
        _, ext = os.path.splitext(filepath)
        if ext.lower() == ".json":
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif ext.lower() in [".yaml", ".yml"]:
            with open(filepath, "w") as f:
                yaml.dump(self.to_dict(), f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @classmethod
    def from_file(cls, filepath: str):
        """Load config from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        _, ext = os.path.splitext(filepath)
        if ext.lower() == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)
        elif ext.lower() in [".yaml", ".yml"]:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        return cls.from_dict(data)

    """Configuration for HTSR (Hurst Temperature Spectral Rigidity) monitoring.

    This config enables grokking detection during training by monitoring
    the spectral properties of weight matrices.

    Key thresholds:
    - α > 4.5: Over-grokking / excessive memorization (SEVERE)
    - α variance > 0.5: Unstable training (SEVERE)
    """

    # Enable/disable HTSR monitoring
    enable_htsr_monitoring: bool = False

    # Monitoring frequency (check every N steps)
    htsr_monitor_interval: int = 75

    # α thresholds for grokking detection
    htsr_alpha_upper_threshold: float = 4.5  # Over-grokking threshold
    htsr_variance_threshold: float = 0.5  # High variance threshold

    # Intervention settings
    htsr_lr_reduction_factor: float = 0.5  # Reduce LR by 50%
    htsr_wd_increase_factor: float = 2.0  # Double weight decay
    htsr_apply_intervention: bool = True  # Auto-apply interventions

    # Dashboard settings
    htsr_dashboard_port: int = 8050
    htsr_dashboard_host: str = "0.0.0.0"  # Local network access
    htsr_dashboard_auth: bool = True
    htsr_dashboard_auto_refresh: int = 120  # seconds

    # Communication channels (for severe alerts)
    htsr_comm_email_enabled: bool = False
    htsr_comm_slack_enabled: bool = False
    htsr_comm_discord_enabled: bool = False
    htsr_comm_pagerduty_enabled: bool = False

    # Loss thresholds for alerts
    htsr_train_loss_warning: float = 1.0
    htsr_train_loss_critical: float = 0.1
    htsr_val_loss_warning: float = 1.5
    htsr_val_loss_critical: float = 0.2

HTSRConfig = ModelConfig

@dataclass
class TrainingConfig:
    """Configuration for training"""

    # Basic training
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_steps: int = 100
    save_steps: int = 10
    eval_steps: int = 10

    # Optimizer
    optimizer: str = "adamw"  # "adamw", "lion", "adafactor"
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.075
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
    max_seq_length: int = 1024
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
    distributed_backend: str = "fsdp"  # "ddp", "fsdp"
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = True

    # Monitoring
    profile_memory: bool = True
    profile_time: bool = True

    # RLHF
    rl_stage: int = 2  # 1 for standard reward, 2 for multi-attribute

    # Testing
    use_mock_data: bool = False

    # Pruning
    pruning_ratio: float = 0.1
    pruning_steps: Optional[List[int]] = None

    # Ring Attention
    use_ring_attention: bool = False

    # Enhanced MoE Training Features
    # Expert specialization and monitoring
    num_experts: int = 8
    num_languages: int = 6
    expert_monitor_log_frequency: int = 50
    expert_monitor_save_frequency: int = 500

    # Checkpointing and memory management
    checkpoint_memory_threshold: float = 0.7
    checkpoint_frequency: int = 2
    memory_cleanup_frequency: int = 50
    memory_target: float = 0.8
    enable_dynamic_batching: bool = True

    # Dynamic optimizations
    expert_capacity_factor: float = 1.25

    # Adaptive attention selection
    seq_length_threshold_mla: int = 2048
    seq_length_threshold_dsa: int = 4096
    memory_threshold_mla: float = 0.6

    # Coherence-based scheduler
    coherence_target: float = 0.7
    coherence_adjustment_frequency: int = 50

    # Enhanced TUI
    tui_update_frequency: int = 1
    tui_save_frequency: int = 100
    tui_log_file: str = "./logs/enhanced_training.json"
    tui_show_plots: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate training configuration"""
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate must be positive")
        if self.optimizer not in ["adamw", "lion", "adafactor"]:
            raise ConfigError(f"Unsupported optimizer: {self.optimizer}")
        if self.lr_schedule not in ["cosine", "linear", "constant"]:
            raise ConfigError(f"Unsupported lr_schedule: {self.lr_schedule}")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigError("gradient_accumulation_steps must be positive")
        if self.save_steps <= 0:
            raise ConfigError("save_steps must be positive")

        if self.beta1 < 0 or self.beta1 >= 1:
            raise ConfigError("beta1 must be in [0, 1)")
        if self.beta2 < 0 or self.beta2 >= 1:
            raise ConfigError("beta2 must be in [0, 1)")

        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def get_small_training_config(cls):
        """Returns a minimal training configuration for testing"""
        return cls(
            batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=10,
            save_steps=5,
            eval_steps=5,
            warmup_steps=1,
            log_every_n_steps=1,
            use_8bit_optimizer=False,
            bf16=False,
            profile_memory=False,
            profile_time=False,
        )


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
    max_seq_length: int = 1024
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

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate inference configuration"""
        if self.temperature < 0 or self.temperature > 2.0:
            raise ConfigError("temperature must be between 0 and 2.0")
        if self.top_p < 0 or self.top_p > 1.0:
            raise ConfigError("top_p must be between 0 and 1.0")
        if self.top_k <= 0:
            raise ConfigError("top_k must be positive")
        if self.weight_bits not in [4, 8, 16]:
            raise ConfigError("weight_bits must be 4, 8 or 16")

        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def get_small_inference_config(cls):
        """Returns a minimal inference configuration for testing"""
        return cls(
            max_new_tokens=64,
            do_sample=False,
            top_k=10,
            use_kv_cache=False,
            use_fp8_inference=False,
            max_batch_size=2,
        )
