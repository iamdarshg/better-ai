
import torch
from .unified_dataloader import create_dataloader, UnifiedDataLoader

def custom_collate_fn(batch):
    """Handle batch collation with mixed data types for MoE training"""
    if len(batch) == 0:
        return {}
    
    # Handle case where batch items might be lists or tensors
    collated = {}
    
    # Standard tensor fields
    tensor_fields = ['input_ids', 'attention_mask', 'labels']
    for field in tensor_fields:
        if field in batch[0]:
            values = []
            for item in batch:
                value = item[field]
                if isinstance(value, list):
                    value = torch.tensor(value, dtype=torch.long)
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.long)
                values.append(value)
            collated[field] = torch.stack(values)
    
    # String fields
    string_fields = ['language', 'repo']
    for field in string_fields:
        if field in batch[0]:
            collated[field] = [item[field] for item in batch]
    
    return collated


def create_code_dataloaders(config, tokenizer, batch_size=4, num_workers=0):
    """
    Wrapper around create_dataloader for backward compatibility.
    Uses the new UnifiedDataLoader system.
    """
    print(f"Creating unified dataloaders for {config.get('primary_dataset', 'default')}")
    
    dataset_config = {
        'path': config.get('primary_dataset', 'bigcode/the-stack'),
        'max_seq_length': config.get('max_length', 1024),
        'languages': config.get('languages', ['Python', 'C', 'Rust']),
        'data_format': 'text'
    }
    
    train_dataloader = create_dataloader(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        split='train',
        streaming=config.get('use_streaming', True),
        num_workers=num_workers
    )
    
    # Create a small eval dataloader using the same config but different split if possible
    # Note: the-stack-v1 only has train split usually, so we might just use a subset or different dataset
    eval_dataloader = create_dataloader(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        split='train', # Fallback to train for stack
        streaming=config.get('use_streaming', True),
        num_workers=num_workers
    )
    
    return train_dataloader, eval_dataloader
