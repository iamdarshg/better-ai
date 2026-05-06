"""
Curriculum-aware dataloader for Better AI training pipeline
Integrates with ExtendedCurriculumScheduler for dynamic sequence lengths,
difficulty filtering, and domain mixing.
"""

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
from typing import Union, List, Dict, Any, Optional
from collections import defaultdict
import random

from .unified_dataloader import (
    StreamingDataset,
    CombinedStreamingDataset,
    parse_xml_tags,
)
from ..training.extended_curriculum import ExtendedCurriculumScheduler

logger = logging.getLogger(__name__)


class CurriculumStreamingDataset(StreamingDataset):
    """
    Streaming dataset that respects curriculum constraints:
    - Dynamic sequence lengths from curriculum
    - Difficulty-based filtering
    - Tracks difficulty scores for curriculum updates
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        curriculum_scheduler: Optional[ExtendedCurriculumScheduler] = None,
        base_max_length: int = 8192,
        split: str = "train",
        streaming: bool = True,
        data_format: str = "text",
        languages: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ):
        # Initialize base dataset
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.base_max_length = base_max_length
        self.split = split
        self.streaming = streaming
        self.data_format = data_format
        self.languages = languages
        self.domain = domain

        # Curriculum integration
        self.curriculum_scheduler = curriculum_scheduler
        self.current_max_length = base_max_length

        # Difficulty tracking
        self.difficulty_scores = []
        self.included_samples = 0
        self.filtered_samples = 0

        # Load dataset
        try:
            self.dataset = load_dataset(
                self.dataset_name, split=self.split, streaming=self.streaming
            )
            if self.languages:
                self.dataset = self.dataset.filter(
                    lambda x: x.get("lang") in self.languages
                )
            logger.info(
                f"Loaded curriculum dataset {self.dataset_name} ({self.split} split)"
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise

    def _get_current_max_length(self) -> int:
        """Get current max sequence length from curriculum"""
        if self.curriculum_scheduler:
            return self.curriculum_scheduler.get_dataset_sequence_length(
                self.dataset_name
            )
        return self.base_max_length

    def _should_include_by_difficulty(self, item: Dict[str, Any], text: str) -> bool:
        """Check if sample should be included based on difficulty curriculum"""
        if not self.curriculum_scheduler:
            return True

        # Normalize difficulty score
        seq_length = len(text.split()) if text else 0
        difficulty_score = self.curriculum_scheduler.normalize_difficulty(
            item, seq_length
        )

        # Store for tracking
        self.difficulty_scores.append(difficulty_score)

        # Check if should include
        should_include = self.curriculum_scheduler.should_include_sample(
            difficulty_score
        )

        if should_include:
            self.included_samples += 1
        else:
            self.filtered_samples += 1

        return should_include

    def __iter__(self):
        """Iterate over dataset with curriculum constraints"""
        for item in self.dataset:
            # Update current max length from curriculum
            self.current_max_length = self._get_current_max_length()

            # Extract and format text
            if self.data_format == "text":
                if "text" in item:
                    text = item["text"]
                elif "content" in item:
                    text = item["content"]
                elif "code" in item:
                    text = item["code"]
                else:
                    text = " ".join(str(v) for v in item.values() if isinstance(v, str))

                formatted_text = self._format_with_xml(item)

                # Check difficulty filtering
                if not self._should_include_by_difficulty(item, text):
                    continue

                # Tokenize with dynamic max length
                encoding = self.tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=self.current_max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                # Get difficulty for this sample
                difficulty = 0.5
                if self.curriculum_scheduler:
                    difficulty = self.curriculum_scheduler.normalize_difficulty(
                        item, len(text.split())
                    )

                yield {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze(),
                    "difficulty": torch.tensor(difficulty, dtype=torch.float32),
                    "dataset_name": self.dataset_name,
                    "domain": self.domain or "unknown",
                    "max_length": self.current_max_length,
                }

            elif self.data_format == "rlhf":
                # Handle RLHF format
                chosen = self._format_with_xml(item.get("chosen", {}))
                rejected = self._format_with_xml(item.get("rejected", {}))

                # Check difficulty for both
                if not self._should_include_by_difficulty(
                    item.get("chosen", {}), chosen
                ):
                    continue

                chosen_encoding = self.tokenizer(
                    chosen,
                    truncation=True,
                    max_length=self.current_max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                rejected_encoding = self.tokenizer(
                    rejected,
                    truncation=True,
                    max_length=self.current_max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                difficulty = 0.5
                if self.curriculum_scheduler:
                    difficulty = self.curriculum_scheduler.normalize_difficulty(
                        item, len(chosen.split())
                    )

                yield {
                    "chosen_input_ids": chosen_encoding["input_ids"].squeeze(),
                    "chosen_attention_mask": chosen_encoding[
                        "attention_mask"
                    ].squeeze(),
                    "rejected_input_ids": rejected_encoding["input_ids"].squeeze(),
                    "rejected_attention_mask": rejected_encoding[
                        "attention_mask"
                    ].squeeze(),
                    "difficulty": torch.tensor(difficulty, dtype=torch.float32),
                    "dataset_name": self.dataset_name,
                    "domain": self.domain or "unknown",
                    "max_length": self.current_max_length,
                }

    def get_difficulty_stats(self) -> Dict[str, Any]:
        """Get difficulty filtering statistics"""
        if not self.difficulty_scores:
            return {}

        return {
            "dataset": self.dataset_name,
            "total_samples": len(self.difficulty_scores),
            "included_samples": self.included_samples,
            "filtered_samples": self.filtered_samples,
            "filter_rate": self.filtered_samples / max(1, len(self.difficulty_scores)),
            "avg_difficulty": np.mean(self.difficulty_scores),
            "std_difficulty": np.std(self.difficulty_scores),
            "min_difficulty": np.min(self.difficulty_scores),
            "max_difficulty": np.max(self.difficulty_scores),
        }


class CurriculumDomainMixer:
    """
    Manages domain-based dataset mixing according to curriculum weights.
    Samples from datasets proportional to their domain weights.
    """

    def __init__(
        self,
        dataset_configs: List[Dict[str, Any]],
        tokenizer,
        curriculum_scheduler: Optional[ExtendedCurriculumScheduler] = None,
        split: str = "train",
        streaming: bool = True,
        distributed: bool = False,
    ):
        self.dataset_configs = dataset_configs
        self.tokenizer = tokenizer
        self.curriculum_scheduler = curriculum_scheduler
        self.split = split
        self.streaming = streaming

        # Create datasets grouped by domain
        self.domain_datasets = defaultdict(list)
        self.dataset_map = {}

        for config in dataset_configs:
            dataset_name = config["name"]
            domain = config.get("domain", "default")

            dataset = CurriculumStreamingDataset(
                dataset_name=config["path"],
                tokenizer=tokenizer,
                curriculum_scheduler=curriculum_scheduler,
                base_max_length=config.get("max_seq_length", 8192),
                split=split,
                streaming=streaming,
                data_format=config.get("data_format", "text"),
                languages=config.get("languages"),
                domain=domain,
                distributed=distributed,
            )

            self.domain_datasets[domain].append(dataset)
            self.dataset_map[dataset_name] = dataset

        # Track samples drawn per domain
        self.domain_sample_counts = defaultdict(int)

        logger.info(
            f"Initialized CurriculumDomainMixer with {len(dataset_configs)} datasets in {len(self.domain_datasets)} domains"
        )

    def _get_domain_weights(self) -> Dict[str, float]:
        """Get current domain weights from curriculum"""
        if self.curriculum_scheduler:
            weights = self.curriculum_scheduler.get_domain_weights()
            if weights:
                return weights

        # Default: equal weights
        num_domains = len(self.domain_datasets)
        return {domain: 1.0 / num_domains for domain in self.domain_datasets}

    def _sample_domain(self) -> str:
        """Sample a domain based on current weights"""
        weights = self._get_domain_weights()
        domains = list(self.domain_datasets.keys())

        # Get weights for available domains
        domain_weights = [weights.get(domain, 0.0) for domain in domains]

        # Normalize
        total = sum(domain_weights)
        if total > 0:
            domain_weights = [w / total for w in domain_weights]
        else:
            # Fallback to uniform
            domain_weights = [1.0 / len(domains)] * len(domains)

        return np.random.choice(domains, p=domain_weights)

    def __iter__(self):
        """Iterate over datasets according to domain mixing weights"""
        # Create iterators for all domains
        domain_iters = {}
        for domain, datasets in self.domain_datasets.items():
            # Round-robin within domain
            domain_iters[domain] = self._round_robin_iterator(datasets)

        while True:
            # Sample domain based on weights
            domain = self._sample_domain()

            try:
                sample = next(domain_iters[domain])
                self.domain_sample_counts[domain] += 1
                yield sample
            except StopIteration:
                # This domain is exhausted, remove it
                if domain in domain_iters:
                    del domain_iters[domain]
                    del self.domain_datasets[domain]

                # If no domains left, stop
                if not domain_iters:
                    break

    def _round_robin_iterator(self, datasets: List[CurriculumStreamingDataset]):
        """Round-robin iterate over multiple datasets"""
        iterators = [iter(ds) for ds in datasets]

        while iterators:
            for it in list(iterators):
                try:
                    yield next(it)
                except StopIteration:
                    iterators.remove(it)

    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get domain sampling statistics"""
        total_samples = sum(self.domain_sample_counts.values())

        if total_samples == 0:
            return {}

        return {
            "domain_counts": dict(self.domain_sample_counts),
            "domain_percentages": {
                domain: count / total_samples
                for domain, count in self.domain_sample_counts.items()
            },
            "target_weights": self._get_domain_weights(),
        }


class CurriculumCombinedDataset(IterableDataset):
    """
    Combined curriculum dataset that handles:
    - Dynamic sequence lengths per dataset
    - Difficulty-based filtering
    - Domain mixing with adaptive weights
    """

    def __init__(
        self,
        dataset_configs: List[Dict[str, Any]],
        tokenizer,
        curriculum_scheduler: Optional[ExtendedCurriculumScheduler] = None,
        split: str = "train",
        streaming: bool = True,
        distributed: bool = False,
    ):
        self.curriculum_scheduler = curriculum_scheduler
        self.split = split

        # Add domain info to configs if available from curriculum
        if curriculum_scheduler and curriculum_scheduler.config.domain_config:
            domain_config = curriculum_scheduler.config.domain_config
            domain_map = {}
            for domain, datasets in domain_config.domains.items():
                for ds_name in datasets:
                    domain_map[ds_name] = domain

            # Augment configs with domain info
            for config in dataset_configs:
                if config["name"] in domain_map:
                    config["domain"] = domain_map[config["name"]]

        # Initialize domain mixer
        self.domain_mixer = CurriculumDomainMixer(
            dataset_configs=dataset_configs,
            tokenizer=tokenizer,
            curriculum_scheduler=curriculum_scheduler,
            split=split,
            streaming=streaming,
            distributed=distributed,
        )

        logger.info(
            f"Initialized CurriculumCombinedDataset with {len(dataset_configs)} datasets"
        )

    def __iter__(self):
        """Iterate over mixed datasets"""
        return iter(self.domain_mixer)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "sampling": self.domain_mixer.get_sampling_stats(),
        }

        # Add per-dataset difficulty stats
        difficulty_stats = {}
        for dataset_name, dataset in self.domain_mixer.dataset_map.items():
            difficulty_stats[dataset_name] = dataset.get_difficulty_stats()

        stats["difficulty"] = difficulty_stats
        return stats


def create_curriculum_dataloader(
    dataset_config: Union[Dict[str, Any], List[Dict[str, Any]]],
    tokenizer,
    curriculum_scheduler: Optional[ExtendedCurriculumScheduler] = None,
    batch_size: int = 8,
    split: str = "train",
    streaming: bool = True,
    num_workers: int = 0,
    distributed: bool = False,
) -> DataLoader:
    """
    Create a curriculum-aware dataloader.

    Args:
        dataset_config: Single dataset config or list of configs
        tokenizer: Tokenizer to use
        curriculum_scheduler: ExtendedCurriculumScheduler instance
        batch_size: Batch size
        split: Dataset split (train/test)
        streaming: Whether to use streaming mode
        num_workers: Number of data loading workers
        distributed: Whether to use distributed data sharding

    Returns:
        DataLoader with curriculum-aware sampling
    """
    if isinstance(dataset_config, list):
        # Multiple datasets - use domain mixing
        dataset = CurriculumCombinedDataset(
            dataset_configs=dataset_config,
            tokenizer=tokenizer,
            curriculum_scheduler=curriculum_scheduler,
            split=split,
            streaming=streaming,
            distributed=distributed,
        )
    else:
        # Single dataset
        dataset = CurriculumStreamingDataset(
            dataset_name=dataset_config["path"],
            tokenizer=tokenizer,
            curriculum_scheduler=curriculum_scheduler,
            base_max_length=dataset_config.get("max_seq_length", 8192),
            split=split,
            streaming=streaming,
            data_format=dataset_config.get("data_format", "text"),
            languages=dataset_config.get("languages"),
            distributed=distributed,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_curriculum_from_datasets_yml(
    stage: str,
    config_path: str = "datasets.yml",
) -> Dict[str, Any]:
    """
    Load curriculum configuration from datasets.yml

    Args:
        stage: Training stage (pretraining, sft, rlhf, security_dpo)
        config_path: Path to datasets.yml

    Returns:
        Curriculum configuration dict
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    curriculum_config = config.get("curriculum", {})
    stages_config = curriculum_config.get("stages", {})

    if stage not in stages_config:
        logger.warning(f"No curriculum configuration found for stage {stage}")
        return {}

    return stages_config[stage]
