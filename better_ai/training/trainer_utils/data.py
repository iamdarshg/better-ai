
import torch
from typing import Dict, Any

def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Process batch with type safety (fix for collation error)"""
    processed_batch = {}

    for key, value in batch.items():
        if hasattr(value, 'to') and callable(getattr(value, 'to')):
            # Tensor - move to device
            processed_batch[key] = value.to(self.device)
        elif isinstance(value, list):
            # List of strings or other non-tensor data
            processed_batch[key] = value
        else:
            # Try to convert to tensor if possible
            try:
                if isinstance(value, (int, float)):
                    processed_batch[key] = torch.tensor([value], dtype=torch.long, device=self.device)
                else:
                    processed_batch[key] = value
            except Exception:
                processed_batch[key] = value

    return processed_batch
