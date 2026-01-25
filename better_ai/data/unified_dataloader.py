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
    """A streaming dataset that can handle any dataset from Hugging Face"""

    def __init__(self, dataset_name, tokenizer, max_length=8192, split="train", streaming=True, data_format="text"):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.streaming = streaming
        self.data_format = data_format

        try:
            self.dataset = load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)
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

                text = parse_xml_tags(text)

                encoding = self.tokenizer(
                    text,
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
                chosen = item["chosen"]
                rejected = item["rejected"]

                chosen = parse_xml_tags(chosen)
                rejected = parse_xml_tags(rejected)

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

def create_dataloader(
    dataset_name,
    tokenizer_name="microsoft/CodeGPT-small-py",
    batch_size=8,
    max_length=8192,
    split="train",
    streaming=True,
    num_workers=0,
    data_format="text",
    **kwargs
):
    """Create a dataloader for a streaming dataset"""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = StreamingDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=max_length,
        split=split,
        streaming=streaming,
        data_format=data_format,
        **kwargs
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

    dataloader = create_dataloader(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        batch_size=2,
        max_length=1024
    )

    for batch in islice(dataloader, 2):
        print("Input IDs:", batch["input_ids"].shape)
        print("Attention Mask:", batch["attention_mask"].shape)
        print("Labels:", batch["labels"].shape)
        print("-" * 20)
