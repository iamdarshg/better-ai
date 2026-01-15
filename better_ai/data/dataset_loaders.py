"""
Dataset loading utilities for Better AI training pipeline
Supports all specified datasets: Stack v2, Magicoder, Code-Feedback, CodeUltraFeedback, RLVR, SWE-bench
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import os
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class StackV2Dataset(Dataset):
    """The Stack v2 dataset for pretraining"""
    
    def __init__(
        self,
        split: str = "train",
        max_length: int = 8192,
        tokenizer_name: str = "microsoft/CodeGPT-small-py",
        languages: Optional[List[str]] = None,
        streaming: bool = True
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load Stack v2 dataset - no fallback, must have real dataset access
        try:
            self.dataset = load_dataset("bigcode/the-stack-v2", split=split, streaming=streaming)
            logger.info(f"Loaded Stack v2 {split} split")
        except Exception as e:
            # Handle case where a requested split (e.g., 'validation') does not exist
            msg = str(e)
            logger.error(f"Failed to load Stack v2 dataset: {msg}. Trying fallback if appropriate.")

            # If it's a Bad split error, attempt to fall back to 'train'
            if "Bad split" in msg and "Available splits" in msg:
                logger.warning(
                    f"Requested split '{split}' not available for bigcode/the-stack-v2. Falling back to 'train' split."
                )
                try:
                    self.dataset = load_dataset("bigcode/the-stack-v2", split="train", streaming=streaming)
                    logger.info("Loaded Stack v2 train split as fallback")
                except Exception as e2:
                    logger.error(f"Fallback to 'train' also failed: {e2}")
                    raise RuntimeError(
                        f"Stack v2 dataset unavailable: {e2}. Please ensure Hugging Face authentication is configured."
                    )
            else:
                # Unknown error - re-raise as runtime error with guidance
                raise RuntimeError(
                    f"Stack v2 dataset unavailable: {e}. Please ensure Hugging Face authentication is configured."
                )
        
        # Filter by languages if specified
        if languages:
            self.dataset = self.dataset.filter(lambda x: x.get("language", "python") in languages)
    
    def __len__(self):
        # For streaming datasets, return a large number
        return 1000000
    
    def __getitem__(self, idx):
        # Get next item from streaming dataset
        item = next(iter(self.dataset.skip(idx)))
        
        # Extract code content
        if "content" in item:
            code = item["content"]
        elif "text" in item:
            code = item["text"]
        else:
            code = str(item)
        
        # Tokenize
        encoding = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


class MagicoderDataset(Dataset):
    """Magicoder dataset for supervised fine-tuning"""
    
    def __init__(
        self,
        dataset_name: str = "ise-uiuc/Magicoder-OSS-Instruct-75K",
        split: str = "train",
        max_length: int = 8192,
        tokenizer_name: str = "microsoft/CodeGPT-small-py"
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load Magicoder dataset
        self.dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded Magicoder {dataset_name} {split} split with {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Format instruction-response pair
        if "instruction" in item and "response" in item:
            prompt = f"Instruction: {item['instruction']}\n\nResponse: {item['response']}"
        elif "problem" in item and "solution" in item:
            prompt = f"Problem: {item['problem']}\n\nSolution: {item['solution']}"
        else:
            prompt = str(item)
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


class CodeFeedbackDataset(Dataset):
    """Code-Feedback dataset for multi-turn coding conversations"""

    def __init__(
        self,
        dataset_name: str = "HuggingFaceH4/Code-Feedback",
        split: str = "train_sft",
        max_length: int = 8192,
        tokenizer_name: str = "microsoft/CodeGPT-small-py"
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load Code-Feedback dataset
        self.dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded Code-Feedback {split} split with {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract conversation
        messages = item.get("messages", [])
        
        # Format conversation
        conversation = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            conversation += f"{role}: {content}\n\n"
        
        # Tokenize
        encoding = self.tokenizer(
            conversation.strip(),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


class CodeUltraFeedbackDataset(Dataset):
    """CodeUltraFeedback dataset for RLHF preference pairs"""
    
    def __init__(
        self,
        dataset_name: str = "coseal/CodeUltraFeedback",
        split: str = "train",
        max_length: int = 8192,
        tokenizer_name: str = "microsoft/CodeGPT-small-py"
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load CodeUltraFeedback dataset
        self.dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded CodeUltraFeedback {split} split with {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract instruction and responses
        instruction = item.get("instruction", "")
        responses = item.get("responses", [])
        annotations = item.get("annotations", [])
        
        # Find best and worst responses based on annotations
        if annotations and responses:
            # Sort responses by score
            scored_responses = []
            for i, (response, annotation) in enumerate(zip(responses, annotations)):
                score = annotation.get("rating", 0)
                scored_responses.append((score, response, i))
            
            scored_responses.sort(key=lambda x: x[0], reverse=True)
            
            # Get chosen (best) and rejected (worst) responses
            chosen_response = scored_responses[0][1].get("content", "")
            rejected_response = scored_responses[-1][1].get("content", "")
        else:
            # Fallback: use first two responses
            chosen_response = responses[0].get("content", "") if responses else ""
            rejected_response = responses[1].get("content", "") if len(responses) > 1 else ""
        
        # Format prompt-response pairs
        chosen_prompt = f"Instruction: {instruction}\n\nResponse: {chosen_response}"
        rejected_prompt = f"Instruction: {instruction}\n\nResponse: {rejected_response}"
        
        # Tokenize chosen
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize rejected
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze()
        }


class RLVRCodingDataset(Dataset):
    """RLVR Coding dataset for reasoning traces"""
    
    def __init__(
        self,
        dataset_name: str = "allenai/rlvr-code-data-python-r1-format-filtered",
        split: str = "train",
        max_length: int = 8192,
        tokenizer_name: str = "microsoft/CodeGPT-small-py"
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load RLVR dataset
        self.dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded RLVR Coding {split} split with {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract messages and reasoning output
        messages = item.get("messages", [])
        reasoning_output = item.get("reasoning-output", "")
        
        # Format with reasoning trace
        conversation = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            conversation += f"{role}: {content}\n\n"
        
        # Add reasoning output
        if reasoning_output:
            conversation += f"reasoning: {reasoning_output}\n\n"
        
        # Tokenize
        encoding = self.tokenizer(
            conversation.strip(),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
            "has_reasoning": bool(reasoning_output)
        }


class SWEBenchDataset(Dataset):
    """SWE-bench dataset for software engineering tasks"""
    
    def __init__(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench",
        split: str = "train",
        max_length: int = 8192,
        tokenizer_name: str = "microsoft/CodeGPT-small-py"
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load SWE-bench dataset
        self.dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded SWE-bench {split} split with {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract problem statement and patch
        problem_statement = item.get("problem_statement", "")
        patch = item.get("patch", "")
        
        # Format as software engineering task
        prompt = f"Problem: {problem_statement}\n\nFix: {patch}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
            "instance_id": item.get("instance_id", ""),
            "repo": item.get("repo", "")
        }


def create_dataloader(
    dataset_type: str,
    split: str = "train",
    batch_size: int = 8,
    max_length: int = 8192,
    tokenizer_name: str = "microsoft/CodeGPT-small-py",
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create dataloader for specified dataset type"""

    dataset_classes = {
        "stack_v2": StackV2Dataset,
        "magicoder": MagicoderDataset,
        "code_feedback": CodeFeedbackDataset,
        "code_ultrafeedback": CodeUltraFeedbackDataset,
        "rlvr_coding": RLVRCodingDataset,
        "swe_bench": SWEBenchDataset
    }

    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataset_class = dataset_classes[dataset_type]

    # Force single-threaded for streaming datasets
    if dataset_type == "stack_v2":
        num_workers = 0
        logger.warning("Using num_workers=0 for streaming StackV2Dataset")

    dataset = dataset_class(
        split=split,
        max_length=max_length,
        tokenizer_name=tokenizer_name,
        **kwargs
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_dataset_info(dataset_type: str) -> Dict[str, Any]:
    """Get information about a dataset type"""
    
    dataset_info = {
        "stack_v2": {
            "name": "The Stack v2",
            "description": "Large code dataset for pretraining",
            "splits": ["train"],
            "use_case": "pretraining"
        },
        "magicoder": {
            "name": "Magicoder",
            "description": "OSS instruction-response pairs",
            "splits": ["train"],
            "use_case": "supervised_fine_tuning"
        },
        "code_feedback": {
            "name": "Code-Feedback",
            "description": "Multi-turn coding conversations",
            "splits": ["train"],
            "use_case": "supervised_fine_tuning"
        },
        "code_ultrafeedback": {
            "name": "CodeUltraFeedback",
            "description": "Ranked coding responses for RLHF",
            "splits": ["train"],
            "use_case": "rlhf_preference"
        },
        "rlvr_coding": {
            "name": "RLVR Coding",
            "description": "Reasoning traces for coding problems",
            "splits": ["train"],
            "use_case": "reasoning_training"
        },
        "swe_bench": {
            "name": "SWE-bench",
            "description": "Software engineering bug fixes",
            "splits": ["train", "test"],
            "use_case": "evaluation"
        }
    }
    
    return dataset_info.get(dataset_type, {})
