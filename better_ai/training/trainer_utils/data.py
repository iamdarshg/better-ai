
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Process batch with type safety and enhanced error handling"""
    processed_batch = {}

    for key, value in batch.items():
        try:
            if hasattr(value, 'to') and callable(getattr(value, 'to')):
                # Tensor - move to device
                if hasattr(value, 'shape') and len(value.shape) > 0:
                    # Validate tensor dimensions
                    if any(dim <= 0 for dim in value.shape):
                        logger.warning(f"Invalid tensor dimensions for {key}: {value.shape}")
                        continue
                processed_batch[key] = value.to(self.device)
            elif isinstance(value, list):
                # List of strings or other non-tensor data
                if key in ['input_ids', 'labels', 'attention_mask']:
                    # Convert lists to tensors for critical fields
                    if value and isinstance(value[0], (int, float)):
                        processed_batch[key] = torch.tensor(value, dtype=torch.long, device=self.device)
                    else:
                        processed_batch[key] = value
                else:
                    processed_batch[key] = value
            else:
                # Try to convert to tensor if possible
                if isinstance(value, (int, float)):
                    processed_batch[key] = torch.tensor([value], dtype=torch.long, device=self.device)
                elif isinstance(value, str):
                    # Skip string values that can't be converted
                    processed_batch[key] = value
                else:
                    processed_batch[key] = value

        except Exception as e:
            logger.warning(f"Error processing batch key '{key}': {e}")
            # Skip problematic keys
            continue

    # Validate critical batch fields
    if 'input_ids' in processed_batch:
        input_ids = processed_batch['input_ids']
        if hasattr(input_ids, 'shape') and len(input_ids.shape) > 0:
            if input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0:
                logger.error(f"Invalid input_ids shape: {input_ids.shape}")
                raise ValueError(f"Invalid input_ids shape: {input_ids.shape}")
    
    if 'labels' in processed_batch:
        labels = processed_batch['labels']
        if hasattr(labels, 'shape') and len(labels.shape) > 0:
            if labels.shape[0] <= 0 or labels.shape[1] <= 0:
                logger.error(f"Invalid labels shape: {labels.shape}")
                raise ValueError(f"Invalid labels shape: {labels.shape}")

    return processed_batch
