"""Data loading and processing utilities for DeepSeek model"""

from .hf_datasets import CodeDataset, MixedCodeDataset, create_code_dataloaders
from .curation import (
    DatasetCurator,
    AgentFLANDecomposer,
    curate_training_corpus,
)

try:
    from .curriculum_dataloader import (
        CurriculumStreamingDataset,
        CurriculumDomainMixer,
        CurriculumCombinedDataset,
        create_curriculum_dataloader,
        load_curriculum_from_datasets_yml,
    )

    CURRICULUM_DATALOADER_AVAILABLE = True
except ImportError:
    CurriculumStreamingDataset = None
    CurriculumDomainMixer = None
    CurriculumCombinedDataset = None
    create_curriculum_dataloader = None
    load_curriculum_from_datasets_yml = None
    CURRICULUM_DATALOADER_AVAILABLE = False

try:
    from .dataset_config import load_datasets_by_stage, load_dataset_from_config

    DATASET_CONFIG_AVAILABLE = True
except ImportError:
    load_datasets_by_stage = None
    load_dataset_from_config = None
    DATASET_CONFIG_AVAILABLE = False

__all__ = [
    "CodeDataset",
    "MixedCodeDataset",
    "create_code_dataloaders",
    "DatasetCurator",
    "AgentFLANDecomposer",
    "curate_training_corpus",
    # Curriculum dataloader components
    "CurriculumStreamingDataset",
    "CurriculumDomainMixer",
    "CurriculumCombinedDataset",
    "create_curriculum_dataloader",
    "load_curriculum_from_datasets_yml",
    "CURRICULUM_DATALOADER_AVAILABLE",
    # Dataset config
    "load_datasets_by_stage",
    "load_dataset_from_config",
    "DATASET_CONFIG_AVAILABLE",
]
