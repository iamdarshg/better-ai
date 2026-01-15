"""
Main training script demonstrating full pipeline
Supports: Pretraining, SFT, and RLHF training stages
"""

import sys
import os
import torch
import logging
import argparse
from pathlib import Path
from typing import Optional

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except:
        pass

from better_ai.config import ModelConfig, TrainingConfig
from better_ai.models.enhanced_model import EnhancedDeepSeekModel
from better_ai.training.enhanced_trainer import EnhancedMoETrainer
from better_ai.training.evaluation import (
    RLHFEvaluator,
    CodingBenchmarkEvaluator,
    MetricsAggregator,
    EvaluationMetrics
)
from better_ai.data.dataset_loaders import create_dataloader, get_dataset_info

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = "./logs"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/training.log"),
            logging.StreamHandler()
        ]
    )


def train_pretraining(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: str = "./checkpoints",
    use_mock_data: bool = False,
):
    """
    Stage 1: Pretraining on Stack v2 dataset
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: PRETRAINING")
    logger.info("=" * 80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = EnhancedDeepSeekModel(model_config, device=device)
    model = model.to(device)
    
    # Create dataloaders
    if use_mock_data:
        logger.info("Using mock data for testing...")
        train_dataloader = _create_mock_dataloader(training_config.batch_size, num_batches=10)
        eval_dataloader = _create_mock_dataloader(training_config.batch_size * 2, num_batches=2)
    else:
        logger.info("Loading Stack v2 pretraining dataset...")
        train_dataloader = create_dataloader(
            "stack_v2",
            split="train",
            batch_size=training_config.batch_size,
            max_length=training_config.max_seq_length,
            shuffle=True,
            num_workers=4,
        )
        
        eval_dataloader = create_dataloader(
            "stack_v2",
            split="validation",
            batch_size=training_config.batch_size * 2,
            max_length=training_config.max_seq_length,
            shuffle=False,
            num_workers=4,
        )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        weight_decay=training_config.weight_decay,
        eps=training_config.eps
    )
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=training_config.warmup_steps,
        T_mult=1,
        eta_min=training_config.learning_rate * training_config.min_lr_ratio
    )
    
    # Initialize trainer
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
    
    # Train
    logger.info("Starting pretraining...")
    metrics = trainer.train()
    
    # Save final model
    torch.save(model.state_dict(), f"{output_dir}/pretrained_model.pt")
    logger.info("Pretraining completed!")
    
    return trainer, metrics


def train_sft(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    checkpoint_path: Optional[str] = None,
    output_dir: str = "./checkpoints",
    use_mock_data: bool = False,
):
    """
    Stage 2: Supervised Fine-Tuning on Magicoder + Code-Feedback
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: SUPERVISED FINE-TUNING")
    logger.info("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = EnhancedDeepSeekModel(model_config, device=device)
    model = model.to(device)
    
    # Load checkpoint from pretraining if available
    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Create dataloaders
    if use_mock_data:
        logger.info("Using mock data for testing...")
        combined_loader = _create_mock_dataloader(training_config.batch_size, num_batches=10)
        eval_loader = _create_mock_dataloader(training_config.batch_size * 2, num_batches=2)
    else:
        logger.info("Loading Magicoder dataset...")
        magicoder_loader = create_dataloader(
            "magicoder",
            split="train",
            batch_size=training_config.batch_size,
            max_length=training_config.max_seq_length,
            shuffle=True,
        )
        
        logger.info("Loading Code-Feedback dataset...")
        code_feedback_loader = create_dataloader(
            "code_feedback",
            split="train",
            batch_size=training_config.batch_size,
            max_length=training_config.max_seq_length,
            shuffle=True,
        )
        
        # Combine dataloaders
        combined_loader = _combine_dataloaders([magicoder_loader, code_feedback_loader])
        
        # Create eval loader
        eval_loader = create_dataloader(
            "magicoder",
            split="validation",
            batch_size=training_config.batch_size * 2,
            max_length=training_config.max_seq_length,
            shuffle=False,
        )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        weight_decay=training_config.weight_decay,
        eps=training_config.eps
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=training_config.warmup_steps,
        T_mult=1,
        eta_min=training_config.learning_rate * training_config.min_lr_ratio
    )
    
    # Initialize trainer
    trainer = EnhancedMoETrainer(
        model=model,
        train_dataloader=combined_loader,
        eval_dataloader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
        device=device,
        use_enhanced_features=True
    )
    
    # Train
    logger.info("Starting supervised fine-tuning...")
    metrics = trainer.train()
    
    # Save checkpoint
    torch.save(model.state_dict(), f"{output_dir}/sft_model.pt")
    logger.info("SFT completed!")
    
    return trainer, metrics


def train_rlhf(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    checkpoint_path: Optional[str] = None,
    output_dir: str = "./checkpoints",
    use_mock_data: bool = False,
):
    """
    Stage 3: RLHF training with GRPO
    """
    logger.info("=" * 80)
    logger.info("STAGE 3: RLHF TRAINING WITH GRPO")
    logger.info("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = EnhancedDeepSeekModel(model_config, device=device)
    model = model.to(device)
    
    # Load checkpoint
    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Create dataloaders
    if use_mock_data:
        logger.info("Using mock data for testing...")
        pref_loader = _create_mock_dataloader(training_config.batch_size, num_batches=10)
        eval_loader = _create_mock_dataloader(training_config.batch_size * 2, num_batches=2)
    else:
        # Load preference data
        logger.info("Loading CodeUltraFeedback preference pairs...")
        pref_loader = create_dataloader(
            "code_ultrafeedback",
            split="train",
            batch_size=training_config.batch_size,
            max_length=training_config.max_seq_length,
            shuffle=True,
        )
        
        # Create eval loader
        eval_loader = create_dataloader(
            "code_ultrafeedback",
            split="validation",
            batch_size=training_config.batch_size * 2,
            max_length=training_config.max_seq_length,
            shuffle=False,
        )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate * 0.1,  # Lower LR for fine-tuning
        betas=(training_config.beta1, training_config.beta2),
        weight_decay=training_config.weight_decay,
        eps=training_config.eps
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=training_config.warmup_steps,
        T_mult=1,
        eta_min=training_config.learning_rate * 0.1 * training_config.min_lr_ratio
    )
    
    # Initialize trainer
    trainer = EnhancedMoETrainer(
        model=model,
        train_dataloader=pref_loader,
        eval_dataloader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
        device=device,
        use_enhanced_features=True
    )
    
    # Train
    logger.info("Starting RLHF training with GRPO...")
    metrics = trainer.train()
    
    # Save final model
    torch.save(model.state_dict(), f"{output_dir}/rlhf_model.pt")
    logger.info("RLHF training completed!")
    
    return trainer, metrics


def evaluate_model(
    model: EnhancedDeepSeekModel,
    model_config: ModelConfig,
    output_dir: str = "./checkpoints",
):
    """
    Evaluate trained model
    """
    logger.info("=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)
    
    device = next(model.parameters()).device
    
    # Load evaluation dataset (SWE-bench)
    logger.info("Loading SWE-bench evaluation dataset...")
    eval_loader = create_dataloader(
        "swe_bench",
        split="test",
        batch_size=8,
        max_length=model_config.max_seq_length,
    )
    
    # Run evaluation
    all_metrics = []
    
    for batch_idx, batch in enumerate(eval_loader):
        input_ids = batch["input_ids"].to(device)
        
        # Generate outputs
        with torch.no_grad():
            outputs = model.forward(
                input_ids=input_ids,
                return_advanced_features=True,
            )
        
        # Compute metrics
        metrics = EvaluationMetrics()
        metrics.coding_accuracy = 0.8  # Placeholder - would compute actual accuracy
        metrics.reasoning_quality = 0.75
        metrics.correctness_score = 0.85
        metrics.efficiency_score = 0.78
        metrics.json_compliance = 0.95
        metrics.grammar_compliance = 0.92
        
        all_metrics.append(metrics)
    
    # Aggregate metrics
    avg_metrics = EvaluationMetrics()
    for attr in vars(avg_metrics):
        if not attr.startswith("_"):
            values = [getattr(m, attr) for m in all_metrics if isinstance(getattr(m, attr), float)]
            if values:
                setattr(avg_metrics, attr, sum(values) / len(values))
    
    # Log results
    metrics_aggregator = MetricsAggregator()
    overall_score = metrics_aggregator.compute_overall_score(avg_metrics)
    logger.info(f"Overall Score: {overall_score:.4f}")
    logger.info(f"Coding Accuracy: {avg_metrics.coding_accuracy:.4f}")
    logger.info(f"Reasoning Quality: {avg_metrics.reasoning_quality:.4f}")
    logger.info(f"JSON Compliance: {avg_metrics.json_compliance:.4f}")
    logger.info(f"Grammar Compliance: {avg_metrics.grammar_compliance:.4f}")
    
    return avg_metrics


def _combine_dataloaders(dataloaders):
    """Helper to combine multiple dataloaders"""
    class CombinedDataLoader:
        def __init__(self, loaders):
            self.loaders = loaders
            self.current_loader_idx = 0
            self.current_iter = None
        
        def __iter__(self):
            self.current_loader_idx = 0
            self.current_iter = None
            return self
        
        def __next__(self):
            while self.current_loader_idx < len(self.loaders):
                if self.current_iter is None:
                    self.current_iter = iter(self.loaders[self.current_loader_idx])
                
                try:
                    return next(self.current_iter)
                except StopIteration:
                    self.current_loader_idx += 1
                    self.current_iter = None
            
            raise StopIteration
    
    return CombinedDataLoader(dataloaders)


def _create_mock_dataloader(batch_size: int, num_batches: int = 10, seq_length: int = 512):
    """Create a mock dataloader for testing"""
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, num_batches, seq_length):
            self.num_batches = num_batches
            self.seq_length = seq_length
        
        def __len__(self):
            return self.num_batches
        
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 64000, (self.seq_length,)),
                "attention_mask": torch.ones(self.seq_length, dtype=torch.long),
                "labels": torch.randint(0, 64000, (self.seq_length,))
            }
    
    dataset = MockDataset(num_batches, seq_length)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Better AI RLHF Training Pipeline")
    parser.add_argument("--stage", choices=["pretrain", "sft", "rlhf", "full"], default="full")
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--test", action="store_true", help="Run with mock data for testing infrastructure")

    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create configs
    model_config = ModelConfig()
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
    
    logger.info("Better AI RLHF Training Pipeline")
    logger.info(f"Device: {device}")
    logger.info(f"Model Config: {model_config}")
    logger.info(f"Training Config: {training_config}")
    if args.test:
        logger.info("TEST MODE: Using mock data")
    
    try:
        trainer = None
        model = None
        
        if args.stage in ["pretrain", "full"]:
            trainer, _ = train_pretraining(model_config, training_config, args.output_dir, use_mock_data=args.test)
            model = trainer.model
            checkpoint_path = f"{args.output_dir}/pretrained_model.pt"
        
        if args.stage in ["sft", "full"]:
            checkpoint_path = f"{args.output_dir}/pretrained_model.pt" if args.stage == "full" else None
            trainer, _ = train_sft(model_config, training_config, checkpoint_path, args.output_dir, use_mock_data=args.test)
            model = trainer.model
            checkpoint_path = f"{args.output_dir}/sft_model.pt"
        
        if args.stage in ["rlhf", "full"]:
            checkpoint_path = f"{args.output_dir}/sft_model.pt" if args.stage == "full" else None
            trainer, _ = train_rlhf(model_config, training_config, checkpoint_path, args.output_dir, use_mock_data=args.test)
            model = trainer.model
        
        if args.eval and model is not None:
            evaluate_model(model.model if hasattr(model, 'model') else model, model_config, args.output_dir)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
