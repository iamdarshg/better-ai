"""Simple data pipeline for training"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
import json
import os
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        stride: int = 256
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize all texts
        self.tokenized_texts = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self.tokenized_texts.append(tokens)
        
        # Create sliding windows
        self.samples = []
        for text_id, tokens in enumerate(self.tokenized_texts):
            for i in range(0, len(tokens), stride):
                chunk = tokens[i:i + max_length]
                if len(chunk) >= max_length // 2:  # Minimum length
                    self.samples.append({
                        'input_ids': chunk,
                        'text_id': text_id,
                        'start_pos': i
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = sample['input_ids']
        
        # Pad to max_length if needed
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        
        # Create attention mask
        attention_mask = [1] * len(sample['input_ids']) + [0] * (self.max_length - len(sample['input_ids']))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(input_ids, dtype=torch.long)  # For language modeling
        }


class CodeDataset(Dataset):
    """Dataset for code generation training"""
    
    def __init__(
        self,
        code_samples: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512
    ):
        self.code_samples = code_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.code_samples)
    
    def __getitem__(self, idx):
        sample = self.code_samples[idx]
        
        # Combine prompt and completion
        prompt = sample.get('prompt', '')
        completion = sample.get('completion', '')
        
        text = f"{prompt}\n{completion}"
        
        # Tokenize
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
        
        # Pad if needed
        if len(tokens) < self.max_length:
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor([1] * len(tokens), dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long)
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """Create dataloader with optimal settings"""
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for training stability
        collate_fn=None  # Use default collation
    )


def load_text_data_from_file(file_path: str) -> List[str]:
    """Load text data from a JSONL file"""
    texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    texts.append(data['text'])
                elif 'content' in data:
                    texts.append(data['content'])
            except json.JSONDecodeError:
                continue
    
    return texts


def load_code_data_from_file(file_path: str) -> List[Dict[str, str]]:
    """Load code data from a JSONL file"""
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                samples.append({
                    'prompt': data.get('prompt', ''),
                    'completion': data.get('completion', ''),
                    'language': data.get('language', 'python')
                })
            except json.JSONDecodeError:
                continue
    
    return samples


def create_synthetic_dataset(
    vocab_size: int = 32000,
    num_samples: int = 1000,
    seq_length: int = 512,
    tokenizer=None
) -> Dataset:
    """Create a synthetic dataset for testing"""
    
    class SyntheticDataset(Dataset):
        def __init__(self, vocab_size, num_samples, seq_length):
            self.vocab_size = vocab_size
            self.num_samples = num_samples
            self.seq_length = seq_length
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Generate random tokens (avoid padding token 0 and use reasonable range)
            input_ids = torch.randint(1, min(self.vocab_size, 10000), (self.seq_length,))
            
            # Create realistic attention mask (not all tokens are always attended to)
            mask_length = torch.randint(self.seq_length // 2, self.seq_length + 1, (1,)).item()
            attention_mask = torch.ones(self.seq_length)
            if mask_length < self.seq_length:
                attention_mask[mask_length:] = 0  # Mask some tokens
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids.clone()
            }
    
    return SyntheticDataset(vocab_size, num_samples, seq_length)


def get_data_collator(config):
    """Create data collator for training"""
    def collator(batch):
        """Collate function for training data"""
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item.get('attention_mask', torch.ones_like(item['input_ids'])) for item in batch]
        
        # Pad sequences to max length in batch
        max_length = min(max(len(seq) for seq in input_ids), config.max_seq_length)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            if len(ids) > max_length:
                # Truncate
                padded_ids = ids[:max_length]
                padded_mask = mask[:max_length]
            else:
                # Pad
                pad_length = max_length - len(ids)
                padded_ids = torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)])
                padded_mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
            
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask),
            'labels': torch.stack(padded_input_ids)  # For language modeling
        }
    
    return collator


# Example usage functions
def create_example_training_setup():
    """Create example training setup"""
    
    # Example configuration
    from ..config import TrainingConfig
    
    config = TrainingConfig(
        batch_size=8,
        max_seq_length=512,
        learning_rate=1e-4,
        max_steps=1000,
        save_steps=100,
        eval_steps=200
    )
    
    # Example tokenizer (would normally load from HuggingFace)
    class SimpleTokenizer:
        def __init__(self, vocab_size=32000):
            self.vocab_size = vocab_size
            self.pad_token_id = 0
        
        def encode(self, text, **kwargs):
            # Simple tokenization for demo
            return [hash(text[i:i+3]) % self.vocab_size for i in range(0, min(len(text), 512), 3)]
    
    tokenizer = SimpleTokenizer()
    
    # Create synthetic dataset
    train_dataset = create_synthetic_dataset(
        vocab_size=32000,
        num_samples=1000,
        seq_length=512
    )
    
    eval_dataset = create_synthetic_dataset(
        vocab_size=32000,
        num_samples=200,
        seq_length=512
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return config, train_dataloader, eval_dataloader, tokenizer


def save_training_config(config, output_dir: str):
    """Save training configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    config_dict = {
        'batch_size': config.batch_size,
        'max_seq_length': config.max_seq_length,
        'learning_rate': config.learning_rate,
        'max_steps': config.max_steps,
        'save_steps': config.save_steps,
        'eval_steps': config.eval_steps,
        'warmup_steps': config.warmup_steps,
        'weight_decay': config.weight_decay,
        'gradient_accumulation_steps': config.gradient_accumulation_steps
    }
    
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_training_config(config_path: str):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return config_dict