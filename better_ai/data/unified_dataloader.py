"""
Unified dataloader for Better AI training pipeline
Supports streaming of any dataset from Hugging Face
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
from itertools import islice
import re
from typing import Union, List, Dict, Any

logger = logging.getLogger(__name__)

def parse_xml_tags(text):
    """Parses XML-style tags and replaces them with special tokens"""
    # This approach preserves the structure of the input by replacing tags with special tokens.
    text = re.sub(r'<context>', '[CONTEXT]', text, flags=re.IGNORECASE)
    text = re.sub(r'</context>', '[/CONTEXT]', text, flags=re.IGNORECASE)
    text = re.sub(r'<problem>', '[PROBLEM]', text, flags=re.IGNORECASE)
    text = re.sub(r'</problem>', '[/PROBLEM]', text, flags=re.IGNORECASE)
    text = re.sub(r'<constraints>', '[CONSTRAINTS]', text, flags=re.IGNORECASE)
    text = re.sub(r'</constraints>', '[/CONSTRAINTS]', text, flags=re.IGNORECASE)
    text = re.sub(r'<examples>', '[EXAMPLES]', text, flags=re.IGNORECASE)
    text = re.sub(r'</examples>', '[/EXAMPLES]', text, flags=re.IGNORECASE)
    return text

class StreamingDataset(IterableDataset):
    """A streaming dataset that can handle any dataset from Hugging Face"""

    def _format_with_xml(self, item: dict) -> str:
        """
        Formats a dataset item with XML-style tags, ensuring they are
        treated as context.
        """
        problem = item.get("problem", "")
        constraints = item.get("constraints", "")
        examples = item.get("examples", "")
        context = item.get("context", "")

        # Fallback to using the entire item as text if specific fields are not present
        if not any([problem, constraints, examples, context]):
            content = item.get("text") or item.get("content", "") or item.get("code", "")
            # If the content already has <tag> format, convert it
            content = parse_xml_tags(content)
            if "[CONTEXT]" not in content:
                return f"[CONTEXT]{content}[/CONTEXT]"
            return content

        # Build formatted string with explicit context markers to prevent prompt injection
        # The model should learn that everything between [CONTEXT] and [/CONTEXT] is background info.
        formatted = ""
        if context:
            formatted += f"[CONTEXT]{context}[/CONTEXT]\n"
        if problem:
            formatted += f"[PROBLEM]{problem}[/PROBLEM]\n"
        if constraints:
            formatted += f"[CONSTRAINTS]{constraints}[/CONSTRAINTS]\n"
        if examples:
            formatted += f"[EXAMPLES]{examples}[/EXAMPLES]\n"

        return formatted.strip()

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Formats multi-turn conversation into a single string"""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            formatted += f"[{role}]{content}[/{role}]\n"
        return formatted.strip()

    def __init__(self, dataset_name, tokenizer, max_length=8192, split="train", streaming=True, data_format="text", languages=None, config_name=None):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.streaming = streaming
        self.data_format = data_format
        self.languages = languages
        self.config_name = config_name

        try:
            self.dataset = load_dataset(self.dataset_name, name=self.config_name, split=self.split, streaming=self.streaming)
            if self.languages:
                self.dataset = self.dataset.filter(lambda x: x.get("lang") in self.languages)
            logger.info(f"Loaded dataset {self.dataset_name} ({self.split} split)")
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise

    def __iter__(self):
        for item in self.dataset:
            # Handle Data Mixing (7.14) - Check for conversation type
            is_multiturn = "messages" in item or "conversations" in item

            if self.data_format == "text":
                if is_multiturn:
                    messages = item.get("messages") or item.get("conversations")
                    text = self._format_conversation(messages)
                elif "text" in item:
                    text = item["text"]
                elif "content" in item:
                    text = item["content"]
                elif "code" in item:
                    text = item["code"]
                else:
                    text = " ".join(str(v) for v in item.values() if isinstance(v, str))

                formatted_text = self._format_with_xml(item)

                encoding = self.tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )

                yield {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze()
                }
            elif self.data_format == "rlhf":
                chosen = self._format_with_xml(item["chosen"])
                rejected = self._format_with_xml(item["rejected"])

                chosen_encoding = self.tokenizer(
                    chosen,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )

                rejected_encoding = self.tokenizer(
                    rejected,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )

                yield {
                    "chosen_input_ids": chosen_encoding["input_ids"].squeeze(),
                    "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(),
                    "rejected_input_ids": rejected_encoding["input_ids"].squeeze(),
                    "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze()
                }

# Alias for compatibility with UNIFIED_TODO.md
UnifiedDataLoader = StreamingDataset


class CombinedStreamingDataset(IterableDataset):
    def __init__(self, dataset_configs, tokenizer, max_length=8192, split="train", streaming=True, data_format="text", languages=None, multi_turn_ratio=0.25):
        self.tokenizer = tokenizer
        self.multi_turn_ratio = multi_turn_ratio

        # Categorize datasets
        self.single_turn_datasets = []
        self.multi_turn_datasets = []

        for config in dataset_configs:
            ds = StreamingDataset(
                dataset_name=config['path'],
                config_name=config.get('config_name'),
                tokenizer=tokenizer,
                max_length=config.get('max_seq_length', max_length),
                split=config.get('split', split),
                streaming=streaming,
                data_format=config.get('data_format', data_format),
                languages=config.get('languages', languages)
            )

            # Use metadata or name to distinguish (simplified)
            if "multi-turn" in config['path'].lower() or "conversation" in config['path'].lower():
                self.multi_turn_datasets.append(ds)
            else:
                self.single_turn_datasets.append(ds)

        # Fallback if only one type exists
        if not self.multi_turn_datasets:
            self.multi_turn_datasets = self.single_turn_datasets
            self.multi_turn_ratio = 0.0
        if not self.single_turn_datasets:
            self.single_turn_datasets = self.multi_turn_datasets
            self.multi_turn_ratio = 1.0

    def __iter__(self):
        import random

        st_iterators = [iter(ds) for ds in self.single_turn_datasets]
        mt_iterators = [iter(ds) for ds in self.multi_turn_datasets]

        while st_iterators or mt_iterators:
            # Sample based on ratio (75/25)
            if random.random() < self.multi_turn_ratio and mt_iterators:
                it = random.choice(mt_iterators)
                try:
                    yield next(it)
                except StopIteration:
                    mt_iterators.remove(it)
            elif st_iterators:
                it = random.choice(st_iterators)
                try:
                    yield next(it)
                except StopIteration:
                    st_iterators.remove(it)
            elif mt_iterators: # Fallback to MT if ST is empty
                it = random.choice(mt_iterators)
                try:
                    yield next(it)
                except StopIteration:
                    mt_iterators.remove(it)
            else:
                break


class MemoryMappedDataset(torch.utils.data.Dataset):
    """Placeholder for memory-mapped dataset implementation"""
    def __init__(self, data_path: str, memory_map: bool = True):
        self.data_path = data_path
        self.memory_map = memory_map
    def __len__(self): return 100
    def __getitem__(self, idx): return {"input_ids": torch.zeros(128)}

class AdaptiveBatchLoader:
    """Dynamically adjusts batch size based on memory usage"""
    def __init__(self, base_batch_size: int, memory_threshold: float = 0.8):
        self.base_batch_size = base_batch_size
        self.memory_threshold = memory_threshold
    def adjust_batch_size(self, current_memory_usage: float) -> int:
        if current_memory_usage > self.memory_threshold:
            return max(1, int(self.base_batch_size * 0.5))
        return self.base_batch_size

def create_dataloader(
    dataset_config: Union[Dict[str, Any], List[Dict[str, Any]]],
    tokenizer,
    batch_size=8,
    split="train",
    streaming=True,
    num_workers=0,
):
    """Create a dataloader from a single or multiple dataset configurations."""

    if isinstance(dataset_config, list):
        dataset = CombinedStreamingDataset(
            dataset_configs=dataset_config,
            tokenizer=tokenizer,
            split=split,
            streaming=streaming,
        )
    else:
        dataset = StreamingDataset(
            dataset_name=dataset_config['path'],
            config_name=dataset_config.get('config_name'),
            tokenizer=tokenizer,
            max_length=dataset_config['max_seq_length'],
            split=split,
            streaming=streaming,
            data_format=dataset_config.get('data_format', 'text'),
            languages=dataset_config.get('languages')
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

if __name__ == '__main__':
    # Example usage:
    tokenizer_name = "microsoft/CodeGPT-small-py"
    dataset_name = "HuggingFaceH4/CodeAlpaca_20K"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataloader = create_dataloader(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=1024
    )

    for batch in islice(dataloader, 2):
        print("Input IDs:", batch["input_ids"].shape)
        print("Attention Mask:", batch["attention_mask"].shape)
        print("Labels:", batch["labels"].shape)
        print("-" * 20)
