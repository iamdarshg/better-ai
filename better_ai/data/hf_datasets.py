
import torch
from torch.utils.data import DataLoader
from .datasets.code_dataset import CodeDataset
from .datasets.mixed_code_dataset import MixedCodeDataset
from .datasets.rolling_window_dataset import RollingWindowCodeDataset
from .datasets.expert_aware_dataset import ExpertAwareRollingDataset


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
    """Create dataloaders for code datasets with rolling window support"""
    
    # Rolling window configuration
    use_rolling_windows = config.get('use_rolling_windows', True)
    use_streaming = config.get('use_streaming', True)
    expert_aware = config.get('expert_aware', False)
    
    if not use_rolling_windows:
        # Use original implementation
        return _create_legacy_dataloaders(config, tokenizer, batch_size, num_workers)
    
    print(f"Creating rolling window dataloaders:")
    print(f"  Streaming: {use_streaming}")
    print(f"  Expert-aware: {expert_aware}")
    
    # Rolling window parameters
    window_size = config.get('rolling_window_size', 1000)
    step_size = config.get('rolling_step_size', 500)
    overlap = config.get('rolling_overlap', 100)
    max_concurrent = config.get('max_concurrent_samples', 5000)
    memory_cleanup_interval = config.get('memory_cleanup_interval', 100)
    
    # Languages
    languages = config.get('languages', ['Python', 'C', 'Rust'])
    
    # Training dataset
    train_dataset_kwargs = {
        'dataset_name': config.get('primary_dataset', 'bigcode/the-stack'),
        'split': 'train',
        'tokenizer': tokenizer,
        'max_length': config.get('max_length', 1024),
        'languages': languages,
        'window_size': window_size,
        'step_size': step_size,
        'overlap': overlap,
        'max_total_samples': config.get('max_train_samples', 100000),
        'cache_dir': config.get('cache_dir', './cache'),
        'use_streaming': use_streaming,
        'shuffle_buffer_size': config.get('shuffle_buffer_size', 1000),
        'memory_cleanup_interval': memory_cleanup_interval,
        'max_concurrent_samples': max_concurrent
    }
    
    if expert_aware:
        train_dataset = ExpertAwareRollingDataset(
            num_experts=config.get('num_experts', 8),
            experts_per_token=config.get('experts_per_token', 2),
            expert_buffer_size=config.get('expert_buffer_size', 200),
            batch_size=batch_size,
            **train_dataset_kwargs
        )
    else:
        train_dataset = RollingWindowCodeDataset(**train_dataset_kwargs)
    
    # Evaluation dataset (smaller parameters)
    eval_kwargs = train_dataset_kwargs.copy()
    eval_kwargs.update({
        'split': 'train',  # bigcode/the-stack only has train split
        'window_size': min(window_size, 500),
        'step_size': min(step_size, 250),
        'max_total_samples': config.get('max_eval_samples', 10000),
        'max_concurrent_samples': min(max_concurrent, 2000)
    })
    
    if expert_aware:
        eval_dataset = ExpertAwareRollingDataset(
            num_experts=config.get('num_experts', 8),
            experts_per_token=config.get('experts_per_token', 2),
            expert_buffer_size=config.get('expert_buffer_size', 100),
            batch_size=batch_size,
            **eval_kwargs
        )
    else:
        eval_dataset = RollingWindowCodeDataset(**eval_kwargs)
    

    
    # Create dataloaders with appropriate settings
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset doesn't support shuffle
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_dataloader, eval_dataloader


def _create_legacy_dataloaders(config, tokenizer, batch_size=4, num_workers=0):
    """Legacy dataloader creation (original implementation)"""
    print("Using legacy dataloader implementation (non-streaming)")
    
    # Dataset configurations for Python, C, Rust
    dataset_configs = [
        {
            'name': 'bigcode/the-stack',
            'languages': ['Python'],
            'max_samples': config.get('max_python_samples', 500000)
        },
        {
            'name': 'bigcode/the-stack',
            'languages': ['C'],
            'max_samples': config.get('max_c_samples', 300000)
        },
        {
            'name': 'bigcode/the-stack',
            'languages': ['Rust'],
            'max_samples': config.get('max_rust_samples', 200000)
        }
    ]
    
    # Create training dataset
    train_dataset = MixedCodeDataset(
        dataset_configs=dataset_configs,
        tokenizer=tokenizer,
        max_length=config.get('max_length', 1024),
        total_max_samples=config.get('max_train_samples', 100000),
        cache_dir=config.get('cache_dir', './cache')
    )
    
    # Create evaluation dataset (smaller)
    eval_dataset = MixedCodeDataset(
        dataset_configs=[
            {**dc, 'max_samples': min(dc.get('max_samples', 10000), 5000)} 
            for dc in dataset_configs
        ],
        tokenizer=tokenizer,
        max_length=config.get('max_length', 1024),
        total_max_samples=config.get('max_eval_samples', 10000),
        cache_dir=config.get('cache_dir', './cache')
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_dataloader, eval_dataloader
