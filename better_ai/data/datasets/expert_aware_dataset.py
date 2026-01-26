
import torch
from torch.utils.data import IterableDataset
from typing import List, Dict, Optional, Iterator
from datasets import load_dataset
import gc
import random
from .rolling_window_dataset import RollingWindowCodeDataset


class ExpertAwareRollingDataset(RollingWindowCodeDataset):
    """Rolling window dataset with expert-aware data routing for MoE training"""

    def __init__(
        self,
        num_experts: int = 8,
        experts_per_token: int = 2,
        expert_capacity_factor: float = 1.25,
        expert_buffer_size: int = 200,
        load_balance_weight: float = 0.01,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.expert_capacity_factor = expert_capacity_factor
        self.expert_buffer_size = expert_buffer_size
        self.load_balance_weight = load_balance_weight
        self.batch_size = batch_size

        # Expert-specific buffers
        self.expert_buffers = {i: [] for i in range(num_experts)}
        self.expert_usage = {i: 0 for i in range(num_experts)}

        print(f"ExpertAwareRollingDataset initialized:")
        print(f"  Experts: {num_experts}, Active per token: {experts_per_token}")
        print(f"  Expert buffer size: {expert_buffer_size}")
        print(f"  Batch size: {batch_size}")

    def _route_sample_to_experts(self, sample: Dict) -> List[int]:
        """Simple routing based on language hash (in real implementation, use actual gating)"""
        content = sample.get('content', '')
        language = sample.get('language', '')

        # Simple hash-based routing for demonstration
        content_hash = hash(content) % self.num_experts
        lang_hash = hash(language) % self.num_experts

        # Return top-k experts
        scores = [(i, abs(i - content_hash) + abs(i - lang_hash)) for i in range(self.num_experts)]
        scores.sort(key=lambda x: x[1])

        return [expert_id for expert_id, _ in scores[:self.experts_per_token]]

    def __iter__(self) -> Iterator[Dict]:
        """Iterate with expert-aware routing"""
        if self.dataset is None:
            # Use parent synthetic fallback
            for sample in self._create_synthetic_iterator():
                yield sample
            return

        # Reset state
        self.expert_buffers = {i: [] for i in range(self.num_experts)}
        self.samples_processed = 0
        self.window_count = 0

        # Create iterator
        if self.use_streaming:
            # Try to shuffle, fallback to no shuffle for compatibility
            try:
                dataset_iter = iter(self.dataset.shuffle(seed=42))
            except Exception:
                dataset_iter = iter(self.dataset)

        for raw_sample in dataset_iter:
            if self.max_total_samples and self.samples_processed >= self.max_total_samples:
                break

            # Process sample
            processed_sample = self._filter_and_process_sample(raw_sample)
            if processed_sample:
                tokenized_sample = self._tokenize_sample(processed_sample)
                if tokenized_sample:
                    # Route to experts
                    expert_ids = self._route_sample_to_experts(processed_sample)

                    # Add to expert buffers
                    for expert_id in expert_ids:
                        expert_buffer = self.expert_buffers[expert_id]
                        expert_buffer.append(tokenized_sample)
                        self.expert_usage[expert_id] += 1

                        # Maintain buffer size
                        if len(expert_buffer) > self.expert_buffer_size:
                            expert_buffer = expert_buffer[-self.expert_buffer_size:]
                        self.expert_buffers[expert_id] = expert_buffer

                    self.samples_processed += 1

            # Emit batches when experts have enough data
            min_buffer_size = min(len(buf) for buf in self.expert_buffers.values())
            if min_buffer_size >= self.batch_size if hasattr(self, 'batch_size') else 32:
                # Create balanced batch across experts
                batch = []
                for expert_id, buffer in self.expert_buffers.items():
                    samples_to_take = min(len(buffer), self.batch_size // self.num_experts if hasattr(self, 'batch_size') else 4)
                    batch.extend(buffer[-samples_to_take:])

                # Shuffle batch
                random.shuffle(batch)

                # Emit samples
                for sample in batch:
                    yield sample

                # Clear used samples from buffers
                for expert_id in self.expert_buffers:
                    samples_to_remove = min(len(self.expert_buffers[expert_id]), self.batch_size // self.num_experts if hasattr(self, 'batch_size') else 4)
                    self.expert_buffers[expert_id] = self.expert_buffers[expert_id][samples_to_remove:]

                self.window_count += 1

                # Cleanup
                self._cleanup_memory()

                # Progress
                if self.window_count % 5 == 0:
                    print(f"Expert window {self.window_count}, Samples: {self.samples_processed}")
                    print(f"Expert usage: {self.expert_usage}")

        # Emit remaining samples
        remaining_samples = []
        for buffer in self.expert_buffers.values():
            remaining_samples.extend(buffer)

        random.shuffle(remaining_samples)
        for sample in remaining_samples:
            yield sample
