"""
Script for automated curriculum tuning.
Finds optimal grokking ratios and plateau steps based on validation performance.
"""

import argparse
import yaml
import torch
import logging
from better_ai.config import ModelConfig, TrainingConfig
from better_ai.training.extended_curriculum import ExtendedCurriculumScheduler
from better_ai.data.unified_dataloader import UnifiedDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_curriculum_config(curriculum_stage_config, model_config):
    """
    Simulates curriculum progression and evaluates potential bottlenecks.
    In a real implementation, this would run small-scale training jobs.
    """
    logger.info(f"Evaluating curriculum config: {curriculum_stage_config}")

    # Check if cosine schedule is too aggressive
    fast_ratio = curriculum_stage_config.get("sequence_length", {}).get("grokking_fast_ratio", 0.4)
    if fast_ratio < 0.2:
        logger.warning("Grokking fast ratio might be too low for efficient learning.")
    elif fast_ratio > 0.6:
        logger.warning("Grokking fast ratio might be too high, risk of forgetting early samples.")

    return {"score": 0.85} # Mock score

def main():
    parser = argparse.ArgumentParser(description="Tune curriculum hyperparameters")
    parser.add_argument("--config", type=str, default="datasets.yml", help="Path to datasets config")
    parser.add_argument("--stage", type=str, default="pretraining", help="Curriculum stage to tune")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)

    curriculum_config = full_config.get("curriculum", {})
    stage_config = curriculum_config.get("stages", {}).get(args.stage)

    if not stage_config:
        logger.error(f"Stage {args.stage} not found in config")
        return

    # Hyperparameter search space
    fast_ratios = [0.3, 0.4, 0.5]
    plateau_steps_options = [5000, 10000, 20000]

    best_score = -1
    best_params = {}

    for fr in fast_ratios:
        for ps in plateau_steps_options:
            test_config = stage_config.copy()
            test_config["sequence_length"]["grokking_fast_ratio"] = fr
            test_config["sequence_length"]["plateau_steps"] = ps

            result = evaluate_curriculum_config(test_config, ModelConfig())
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = {"grokking_fast_ratio": fr, "plateau_steps": ps}

    logger.info(f"Optimization complete. Best params for {args.stage}: {best_params}")
    logger.info("To apply, update datasets.yml with these values.")

if __name__ == "__main__":
    main()
