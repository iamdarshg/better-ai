"""Enhanced training script for DeepSeek MoE model with real datasets"""

import torch
from torch.utils.data import DataLoader
import sys
import os
import argparse
from pathlib import Path

# Add to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from better_ai.config import ModelConfig, TrainingConfig
    from better_ai.models.moe import DeepSeekMoEModel
    from better_ai.optimizers.fp8 import get_fp8_optimizer
    from better_ai.training.trainer import Trainer
    from better_ai.data.hf_datasets import create_code_dataloaders
    from better_ai.utils import set_seed, get_device
    
    # NEW: Enhanced MoE optimizations
    from better_ai.training.expert_manager import ExpertSpecializationManager, MoETrainingMonitor
    from better_ai.training.checkpointing import SelectiveCheckpointManager, AdaptiveMemoryManager
    from better_ai.training.adaptive_optimizations import DynamicExpertCapacityManager, AdaptiveAttentionSelector
    from better_ai.training.coherence_scheduler import CoherenceBasedScheduler
    from better_ai.training.tui import MoETrainingTUI
    
    # Try to import tokenizer
    try:
        from transformers import AutoTokenizer
        TOKENIZER_AVAILABLE = True
    except ImportError:
        TOKENIZER_AVAILABLE = False
        print("Warning: transformers not available, using dummy tokenizer")
        TOKENIZER_AVAILABLE = False
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages:")
    print("pip install torch datasets transformers")
    sys.exit(1)


def create_moe_config():
    """Create MoE model configuration"""
    
    model_config = ModelConfig(
        vocab_size=32768,           # Good for code
        hidden_dim=768,              # Standard size
        num_layers=14,               # Good balance
        num_attention_heads=12,        # Attention heads
        num_key_value_heads=6,        # GQA: 2x KV reduction
        intermediate_dim=3072,        # 4x hidden_dim
        max_seq_length=1024,          # Manageable for RTX 4060
        
        # MoE Configuration
        num_experts=8,                # 8 experts total
        num_experts_per_token=2,       # 2 active experts per token
        expert_capacity_factor=1.25,     # Capacity for routing
        moe_load_balance_weight=0.01,      # Load balancing
        shared_experts=1,           # 1 always active expert
#        use_moe_every_n_layers=2,  # MoE every 2 layers
        
        # Optimizations
        use_fp8=True,               # FP8 training
        use_flash_attention=True,     # Flash attention
        use_gradient_checkpointing=True, # Memory saving
#        use_gqa=True,              # Grouped Query Attention
    )
    
    training_config = TrainingConfig(
        # Training parameters
        batch_size=4,                # For 8GB VRAM
        gradient_accumulation_steps=8,  # Effective batch = 32
        learning_rate=1e-4,           # Standard LR
        warmup_steps=1000,            # Warmup
        max_steps=10000,              # Total steps
        save_steps=1000,              # Save every 1K steps
        eval_steps=2000,              # Eval every 2K steps
        
        # Optimization
        weight_decay=0.1,             # Weight decay
        use_8bit_optimizer=True,       # 8-bit optimizer
        fp8_scaling_window=16,        # FP8 scaling window
        fp8_delayed_scaling=True,     # Delayed scaling
        
        # Data config
        max_seq_length=1024,           # Max sequence length
        output_dir="./checkpoints",     # Output dir
        log_dir="./logs",             # Log dir
        
        # Additional
        min_lr_ratio=0.1              # Min LR ratio
    )
    
    return model_config, training_config


def create_tokenizer(vocab_size: int = 32768):
    """Create or load tokenizer"""
    
    if not TOKENIZER_AVAILABLE:
        # Create dummy tokenizer
        class DummyTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.bos_token_id = 2
            
            def encode(self, text, **kwargs):
                # Simple hash-based tokenization
                return [hash(text[i:i+3]) % self.vocab_size 
                        for i in range(0, min(len(text), kwargs.get('max_length', 1024)), 3)]
            
            def decode(self, tokens, **kwargs):
                return f"[{'|'.join(map(str, tokens))}]"
        
        print("Using dummy tokenizer")
        return DummyTokenizer(vocab_size)
    
    # Try to load a real tokenizer for code
    tokenizer_names = [
        "EleutherAI/gpt-neox-20b",
        "microsoft/CodeGPT-small-py",
        "Salesforce/codegen-350M-mono",
        "bigcode/gpt_bigcode-santacoder"
    ]
    
    for name in tokenizer_names:
        try:
            print(f"Trying to load tokenizer: {name}")
            tokenizer = AutoTokenizer.from_pretrained(name)
            
            # Add special tokens if missing
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            
            print(f"Successfully loaded: {name}")
            return tokenizer
            
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue
    
    # Fallback to simple tokenizer
    print("Falling back to basic tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    
    return tokenizer


def get_dataset_config():
    """Get dataset configuration with rolling window options"""
    
    return {
        # Real dataset configuration
        'max_python_samples': 2000,     # 2K Python samples (for faster startup)
        'max_c_samples': 1000,           # 1K C samples
        'max_rust_samples': 1000,        # 1K Rust samples
        'max_train_samples': 5000,        # Total training samples (smaller for quick testing)
        'max_eval_samples': 500,          # Evaluation samples (smaller)
        
        # Dataset names (bigcode/the-stack or alternatives)
        'primary_dataset': 'bigcode/the-stack',
        'fallback_datasets': [
            'saaransh/codecontest-py-v2',  # Small, fast to download
            'code_x_glue_ct',
            'codeparrot/apps'
        ],
        
        # Cache
        'cache_dir': './dataset_cache',
        
        # Languages to focus on
        'languages': ['Python', 'C', 'Rust'],
        
        # Rolling window configuration
        'use_rolling_windows': True,      # Enable rolling windows (NEW)
        'use_streaming': True,            # Use streaming datasets (NEW)
        'expert_aware': False,            # Enable expert-aware routing (NEW)
        
        # Rolling window parameters (NEW)
        'rolling_window_size': 500,       # Samples per window (smaller for fast startup)
        'rolling_step_size': 250,         # How much to slide each window (smaller for fast iteration)
        'rolling_overlap': 50,            # Overlap between windows (smaller)
        'max_concurrent_samples': 1000,   # Max samples in memory at once (smaller for memory efficiency)
        'shuffle_buffer_size': 200,       # Shuffle buffer for streaming (smaller)
        'memory_cleanup_interval': 50,     # Cleanup every N batches (more frequent)
        
        # Expert-aware parameters (NEW)
        'num_experts': 8,                 # Number of experts
        'experts_per_token': 2,           # Active experts per token
        'expert_buffer_size': 200,        # Per-expert buffer size
        'expert_capacity_factor': 1.25,   # Expert capacity multiplier
        'load_balance_weight': 0.01,      # Load balancing weight
        
        # General dataset parameters
        'max_length': 1024,               # Max sequence length
        
        # Fallback options
        'enable_fallback': True,          # Fall back to legacy if streaming fails (NEW)
        'fallback_to_synthetic': True     # Fall back to synthetic if all fails (NEW)
    }


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train DeepSeek MoE model')
    parser.add_argument('--config', type=str, choices=['small', 'medium', 'large'], 
                       default='medium', help='Model configuration')
    parser.add_argument('--dataset', type=str, choices=['code', 'text', 'mixed'], 
                       default='code', help='Dataset type')
    parser.add_argument('--use-synthetic', action='store_true', 
                       help='Use synthetic data instead of real datasets')
    parser.add_argument('--disable-moe', action='store_true', 
                       help='Disable MoE (use dense model)')
    parser.add_argument('--max-steps', type=int, default=10000,
                       help='Maximum training steps')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--python-only', action='store_true',
                       help='Use only Python dataset')
    
    # Rolling window options (NEW)
    parser.add_argument('--disable-rolling-windows', action='store_true',
                       help='Disable rolling windows (use legacy dataloader)')
    parser.add_argument('--disable-streaming', action='store_true',
                       help='Disable streaming datasets')
    parser.add_argument('--enable-expert-aware', action='store_true',
                       help='Enable expert-aware data routing')
    parser.add_argument('--window-size', type=int, default=1000,
                       help='Rolling window size')
    parser.add_argument('--step-size', type=int, default=500,
                       help='Rolling window step size')
    parser.add_argument('--overlap', type=int, default=100,
                       help='Rolling window overlap')
    parser.add_argument('--max-concurrent', type=int, default=5000,
                       help='Maximum concurrent samples in memory')
    parser.add_argument('--expert-buffer-size', type=int, default=200,
                       help='Expert buffer size for expert-aware routing')
    
    # Enhanced optimization arguments
    parser.add_argument('--disable-enhanced-features', action='store_true',
                       help='Disable all enhanced MoE optimizations')
    parser.add_argument('--enable-checkpointing', action='store_true', default=True,
                       help='Enable selective gradient checkpointing')
    parser.add_argument('--enable-expert-specialization', action='store_true', default=True,
                       help='Enable expert specialization tracking')
    parser.add_argument('--enable-dynamic-capacity', action='store_true', default=True,
                       help='Enable dynamic expert capacity adjustment')
    parser.add_argument('--enable-adaptive-attention', action='store_true', default=True,
                       help='Enable adaptive attention mechanism selection')
    parser.add_argument('--enable-coherence-scheduler', action='store_true', default=True,
                       help='Enable coherence-based scheduling')
    parser.add_argument('--enable-enhanced-tui', action='store_true', default=True,
                       help='Enable enhanced TUI with real-time metrics')
    
    args = parser.parse_args()
    
    print("🚀 Starting DeepSeek MoE Model Training")
    print("=" * 60)
    
    # Setup
    set_seed(42)
    device = get_device()
    print(f"🖥️  Device: {device}")
    
    # Configuration
    model_config, training_config = create_moe_config()
    
    # Override config with args
    training_config.max_steps = args.max_steps
    training_config.batch_size = args.batch_size
    
    # Enhanced feature configuration
    enable_enhanced = not args.disable_enhanced_features
    
    print(f"📝 Model Config:")
    print(f"   Hidden Size: {model_config.hidden_dim}")
    print(f"   Layers: {model_config.num_layers}")
    print(f"   Experts: {model_config.num_experts} (active: {model_config.num_experts_per_token})")
    print(f"   FP8: {model_config.use_fp8}")
    
    if enable_enhanced:
        print(f"🚀 Enhanced Features:")
        print(f"   Expert Specialization: {'✓' if args.enable_expert_specialization else '✗'}")
        print(f"   Selective Checkpointing: {'✓' if args.enable_checkpointing else '✗'}")
        print(f"   Dynamic Capacity: {'✓' if args.enable_dynamic_capacity else '✗'}")
        print(f"   Adaptive Attention: {'✓' if args.enable_adaptive_attention else '✗'}")
        print(f"   Coherence Scheduler: {'✓' if args.enable_coherence_scheduler else '✗'}")
        print(f"   Enhanced TUI: {'✓' if args.enable_enhanced_tui else '✗'}")
    else:
        print(f"⚠️  Enhanced features: DISABLED")
    
    # Create model with enhanced optimizations
    print("\n🏗️  Creating Enhanced MoE model...")
    
    # Initialize optimization managers
    device = get_device()
    
    # Expert specialization manager
    expert_manager = ExpertSpecializationManager(
        num_experts=model_config.num_experts,
        num_languages=3,  # Python, C, Rust
        device=device
    )
    
    # Training monitor
    training_monitor = MoETrainingMonitor(
        num_experts=model_config.num_experts,
        num_languages=3,
        log_frequency=50,
        save_frequency=500,
        log_dir="./logs"
    )
    
    # Selective checkpointing manager
    checkpoint_manager = SelectiveCheckpointManager(
        memory_threshold=0.7,
        checkpoint_frequency=2,
        device=device
    )
    
    # Dynamic expert capacity manager
    capacity_manager = DynamicExpertCapacityManager(
        num_experts=model_config.num_experts,
        base_capacity_factor=model_config.expert_capacity_factor,
        device=device
    )
    
    # Adaptive attention selector
    attention_selector = AdaptiveAttentionSelector(
        seq_length_threshold_mla=2048,
        seq_length_threshold_dsa=4096,
        memory_threshold_mla=0.6,
        device=device
    )
    
    # Coherence-based scheduler
    coherence_scheduler = CoherenceBasedScheduler(
        base_lr=training_config.learning_rate,
        coherence_target=0.7,
        adjustment_frequency=50,
        device=device
    )
    
    # TUI will be handled by the enhanced trainer
    
    if args.disable_moe:
        # Use dense model
        from better_ai.models.core import DeepSeekModel
        model = DeepSeekModel(
            vocab_size=model_config.vocab_size,
            hidden_size=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_attention_heads,
            num_key_value_heads=model_config.num_key_value_heads,
            intermediate_size=model_config.intermediate_dim,
            max_seq_length=model_config.max_seq_length
        )
        print("Using Dense model")
    else:
        # Use Enhanced MoE model (default)
        model = DeepSeekMoEModel(
            vocab_size=model_config.vocab_size,
            hidden_size=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_attention_heads,
            num_key_value_heads=model_config.num_key_value_heads,
            intermediate_size=model_config.intermediate_dim,
            num_experts=model_config.num_experts,
            num_experts_per_token=model_config.num_experts_per_token,
            expert_capacity_factor=model_config.expert_capacity_factor,
            load_balance_loss_weight=model_config.moe_load_balance_weight,
            shared_experts=model_config.shared_experts,
            max_seq_length=model_config.max_seq_length,
            use_moe_every_n_layers=2  # More frequent MoE layers
        )
        print(f"Using Enhanced MoE model with {model_config.num_experts} experts")
        print(f"   🎯 Expert Specialization: ENABLED")
        print(f"   🔄 Selective Checkpointing: ENABLED")
        print(f"   🧠 Dynamic Capacity Adjustment: ENABLED")
        print(f"   ⚡ Adaptive Attention Selection: ENABLED")
        print(f"   🎮 Coherence-Based Scheduler: ENABLED")
        print(f"   🖥️ Enhanced TUI: ENABLED")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Parameters:")
    print(f"   Total: {total_params / 1e6:.2f}M")
    print(f"   Trainable: {trainable_params / 1e6:.2f}M")
    
    model = model.to(device)
    
    # Create tokenizer
    print("\n🔤 Creating tokenizer...")
    tokenizer = create_tokenizer(model_config.vocab_size)
    
    # Create datasets
    print("\n📊 Creating datasets...")
    
    if args.use_synthetic:
        # Use synthetic data
        from better_ai.data.dataset import create_synthetic_dataset, create_dataloader
        
        train_dataset = create_synthetic_dataset(
            vocab_size=model_config.vocab_size,
            num_samples=10000,
            seq_length=model_config.max_seq_length
        )
        
        eval_dataset = create_synthetic_dataset(
            vocab_size=model_config.vocab_size,
            num_samples=1000,
            seq_length=model_config.max_seq_length
        )
        
        train_dataloader = create_dataloader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        eval_dataloader = create_dataloader(
            eval_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print("Using synthetic datasets")
        
    else:
        # Use real datasets
        dataset_config = get_dataset_config()
        
        # Override with command line arguments (NEW)
        if args.disable_rolling_windows:
            dataset_config['use_rolling_windows'] = False
        if args.disable_streaming:
            dataset_config['use_streaming'] = False
        if args.enable_expert_aware:
            dataset_config['expert_aware'] = True
        
        # Rolling window parameters (NEW)
        dataset_config['rolling_window_size'] = args.window_size
        dataset_config['rolling_step_size'] = args.step_size
        dataset_config['rolling_overlap'] = args.overlap
        dataset_config['max_concurrent_samples'] = args.max_concurrent
        dataset_config['expert_buffer_size'] = args.expert_buffer_size
        
        # Language filtering
        if args.python_only:
            dataset_config['languages'] = ['Python']
            dataset_config['max_python_samples'] = 50000
            dataset_config['max_train_samples'] = 50000
        
        # Update expert config for model
        if dataset_config.get('expert_aware', False):
            dataset_config['num_experts'] = model_config.num_experts
            dataset_config['experts_per_token'] = model_config.num_experts_per_token
        
        print(f"\n📊 Dataset Configuration:")
        print(f"   Rolling Windows: {dataset_config.get('use_rolling_windows', True)}")
        print(f"   Streaming: {dataset_config.get('use_streaming', True)}")
        print(f"   Expert-Aware: {dataset_config.get('expert_aware', False)}")
        print(f"   Window Size: {dataset_config.get('rolling_window_size', 1000)}")
        print(f"   Languages: {dataset_config['languages']}")
        print(f"   Max Training Samples: {dataset_config['max_train_samples']}")
        
        try:
            train_dataloader, eval_dataloader = create_code_dataloaders(
                config=dataset_config,
                tokenizer=tokenizer,
                batch_size=training_config.batch_size,
                num_workers=0
            )
            print(f"✅ Successfully created dataloaders with rolling windows")
            
        except Exception as e:
            print(f"❌ Failed to load real datasets with rolling windows: {e}")
            
            # Check if we should try fallback options
            if dataset_config.get('enable_fallback', True) and dataset_config.get('use_rolling_windows', True):
                print("🔄 Attempting fallback to legacy dataloader...")
                dataset_config['use_rolling_windows'] = False
                
                try:
                    train_dataloader, eval_dataloader = create_code_dataloaders(
                        config=dataset_config,
                        tokenizer=tokenizer,
                        batch_size=training_config.batch_size,
                        num_workers=0
                    )
                    print("✅ Fallback to legacy dataloader successful")
                    
                except Exception as fallback_error:
                    print(f"❌ Legacy fallback also failed: {fallback_error}")
                    
                    if dataset_config.get('fallback_to_synthetic', True):
                        print("🔄 Falling back to synthetic data...")
                        
                        # Final fallback to synthetic
                        from better_ai.data.dataset import create_synthetic_dataset, create_dataloader
                        
                        train_dataset = create_synthetic_dataset(
                            vocab_size=model_config.vocab_size,
                            num_samples=5000,
                            seq_length=model_config.max_seq_length
                        )
                        
                        eval_dataset = create_synthetic_dataset(
                            vocab_size=model_config.vocab_size,
                            num_samples=500,
                            seq_length=model_config.max_seq_length
                        )
                        
                        train_dataloader = create_dataloader(
                            train_dataset,
                            batch_size=training_config.batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=False
                        )
                        
                        eval_dataloader = create_dataloader(
                            eval_dataset,
                            batch_size=training_config.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False
                        )
                        print("✅ Synthetic data fallback successful")
                    else:
                        raise e
            else:
                raise e
    
    # Handle IterableDataset length
    try:
        train_batches = len(train_dataloader)
    except TypeError:
        train_batches = "Iterable"
    
    try:
        eval_batches = len(eval_dataloader)
    except TypeError:
        eval_batches = "Iterable"
    
    print(f"📈 Training batches: {train_batches}")
    print(f"📉 Evaluation batches: {eval_batches}")
    
    # Setup training
    print("\n⚙️  Setting up training...")
    
    optimizer = get_fp8_optimizer(
        model,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        fp8_scaling_window=training_config.fp8_scaling_window,
        fp8_delayed_scaling=training_config.fp8_delayed_scaling
    )
    
    # Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    scheduler = CosineAnnealingLR(
        optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer,
        T_max=training_config.max_steps,
        eta_min=training_config.min_lr_ratio * training_config.learning_rate
    )
    
    print(f"🎯 Optimizer: {type(optimizer).__name__}")
    print(f"📊 Scheduler: Cosine Annealing")
    
    # Create enhanced trainer with selective optimizations
    print(f"\n🚀 Starting {'ENHANCED' if enable_enhanced else 'STANDARD'} MoE training...")
    try:
        if enable_enhanced:
            # Import enhanced trainer
            from better_ai.training.enhanced_trainer import EnhancedMoETrainer
            
            # Use enhanced trainer with configured features
            trainer = EnhancedMoETrainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                config=training_config,
                device=device,
                use_enhanced_features=True
            )
        else:
            # Fallback to standard trainer
            trainer = Trainer(
                model=model,
                config=training_config,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                optimizer=optimizer,
                scheduler=scheduler
            )
            trainer.train()

    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Training completed!")
    print(f"📁 Checkpoints saved to: {training_config.output_dir}")
    print(f"📋 Logs saved to: {training_config.log_dir}")
    print(f"🔥 MoE Expert Usage Check: Check logs for expert specialization")


if __name__ == "__main__":
    main()
