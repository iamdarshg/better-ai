
import yaml
from typing import List, Dict, Any

class DatasetConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_dataset_configs(self) -> List[Dict[str, Any]]:
        return self.config.get('datasets', [])

def load_dataset_from_config(name: str, config_path: str = 'datasets.yml'):
    config = DatasetConfig(config_path)
    for dataset_config in config.get_dataset_configs():
        if dataset_config['name'] == name:
            # In a real implementation, this would load the dataset
            # based on the type and path.
            print(f"Loading dataset '{name}' with config: {dataset_config}")
            return dataset_config
    raise ValueError(f"Dataset '{name}' not found in {config_path}")

def load_datasets_by_stage(stage: str, config_path: str = 'datasets.yml') -> List[Dict[str, Any]]:
    config = DatasetConfig(config_path)
    return [
        dataset_config
        for dataset_config in config.get_dataset_configs()
        if dataset_config.get('stage') == stage
    ]

if __name__ == '__main__':
    # Example usage
    dummy_dataset = load_dataset_from_config('dummy')
    another_dummy_dataset = load_dataset_from_config('another_dummy')

    pretraining_datasets = load_datasets_by_stage('pretraining')
    print(f"Pretraining datasets: {pretraining_datasets}")

    sft_datasets = load_datasets_by_stage('sft')
    print(f"SFT datasets: {sft_datasets}")
