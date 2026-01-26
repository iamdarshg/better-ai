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
    text = re.sub(r'<problem>', '[PROBLEM]', text)
    text = re.sub(r'</problem>', '[/PROBLEM]', text)
    text = re.sub(r'<constraints>', '[CONSTRAINTS]', text)
    text = re.sub(r'</constraints>', '[/CONSTRAINTS]', text)
    text = re.sub(r'<examples>', '[EXAMPLES]', text)
    text = re.sub(r'</examples>', '[/EXAMPLES]', text)
    return text

class StreamingDataset(IterableDataset):
    def _format_with_xml(self, item: dict) -> str:
        """Formats a dataset item with XML-style tags."""
        problem = item.get("problem", "")
        constraints = item.get("constraints", "")
        examples = item.get("examples", "")

        # Fallback to using the entire item as text if specific fields are not present
        if not problem and not constraints and not examples:
            return item.get("text") or item.get("content", "")

        return f"<problem>{problem}</problem><constraints>{constraints}</constraints><examples>{examples}</examples>"
    """A streaming dataset that can handle any dataset from Hugging Face"""

    def __init__(self, dataset_name, tokenizer, max_length=8192, split="train", streaming=True, data_format="text", languages=None):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.streaming = streaming
        self.data_format = data_format
        self.languages = languages

        try:
            self.dataset = load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)
            if self.languages:
                self.dataset = self.dataset.filter(lambda x: x.get("lang") in self.languages)
            logger.info(f"Loaded dataset {self.dataset_name} ({self.split} split)")
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise

    def __iter__(self):
        for item in self.dataset:
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


class CombinedStreamingDataset(IterableDataset):
    def __init__(self, dataset_configs, tokenizer, max_length=8192, split="train", streaming=True, data_format="text", languages=None):
        self.datasets = [
            StreamingDataset(
                dataset_name=config['path'],
                tokenizer=tokenizer,
                max_length=config.get('max_seq_length', max_length),
                split=config.get('split', split),
                streaming=streaming,
                data_format=config.get('data_format', data_format),
                languages=config.get('languages', languages)
            )
            for config in dataset_configs
        ]

    def __iter__(self):
        iterators = [iter(ds) for ds in self.datasets]
        while iterators:
            # Iterate over a copy of the list to allow safe removal
            for it in list(iterators):
                try:
                    yield next(it)
                except StopIteration:
                    # This iterator is exhausted, remove it.
                    iterators.remove(it)


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
