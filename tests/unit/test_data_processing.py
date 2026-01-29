"""
Unit tests for data processing and pipeline components
Tests data loading, preprocessing, and pipeline orchestration
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from better_ai.data.unified_dataloader import UnifiedDataLoader, StreamingDataset
from better_ai.data.dataset_config import DatasetConfig, DatasetRegistry
from better_ai.data.hf_datasets import HFDatasetWrapper
from better_ai.training.trainer_utils.data import DataCollator, TokenizedDataset


class TestUnifiedDataLoader:
    """Test unified data loader functionality."""

    def test_dataloader_initialization(self):
        """Test unified data loader initialization."""
        config = {
            "batch_size": 4,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
        }

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        dataloader = UnifiedDataLoader(mock_dataset, **config)

        assert dataloader.batch_size == 4
        assert dataloader.shuffle == True
        assert dataloader.num_workers == 0

    def test_streaming_dataloader(self):
        """Test streaming data loader functionality."""
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([{"text": "test"}] * 10))

        dataloader = UnifiedDataLoader(mock_dataset, batch_size=2, streaming=True)

        # Test streaming behavior
        batches = list(dataloader)
        assert len(batches) == 5  # 10 items / batch_size 2

    def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing based on memory."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        dataloader = UnifiedDataLoader(
            mock_dataset,
            base_batch_size=8,
            adaptive_batching=True,
            memory_threshold=0.8,
        )

        # Simulate high memory usage
        current_memory = 0.9
        adjusted_batch_size = dataloader.adjust_batch_size(current_memory)

        assert adjusted_batch_size < 8

    def test_multi_dataset_loading(self):
        """Test loading from multiple datasets."""
        datasets = {
            "primary": Mock(__len__=Mock(return_value=100)),
            "secondary": Mock(__len__=Mock(return_value=50)),
        }

        dataloader = UnifiedDataLoader(
            datasets, batch_size=4, dataset_weights={"primary": 0.7, "secondary": 0.3}
        )

        assert dataloader.dataset_weights["primary"] == 0.7
        assert dataloader.dataset_weights["secondary"] == 0.3


class TestDatasetConfig:
    """Test dataset configuration and registry."""

    def test_dataset_config_creation(self):
        """Test dataset configuration creation."""
        config = DatasetConfig(
            name="test_dataset",
            path="path/to/dataset",
            split="train",
            batch_size=8,
            max_length=512,
            preprocessing_config={"tokenize": True, "truncate": True, "pad": True},
        )

        assert config.name == "test_dataset"
        assert config.batch_size == 8
        assert config.max_length == 512
        assert config.preprocessing_config["tokenize"] == True

    def test_dataset_registry(self):
        """Test dataset registry functionality."""
        registry = DatasetRegistry()

        # Register dataset
        config = DatasetConfig(name="test", path="test/path")
        registry.register(config)

        # Retrieve dataset
        retrieved = registry.get("test")
        assert retrieved.name == "test"
        assert retrieved.path == "test/path"

        # List all datasets
        all_datasets = registry.list_all()
        assert "test" in all_datasets

    def test_dataset_config_validation(self):
        """Test dataset configuration validation."""
        # Valid config
        config = DatasetConfig(name="valid", path="valid/path", batch_size=4)
        assert config.is_valid()

        # Invalid config (missing required fields)
        with pytest.raises(ValueError):
            DatasetConfig(name="", path="")  # Empty name

    def test_dataset_merging(self):
        """Test merging multiple dataset configurations."""
        config1 = DatasetConfig(name="dataset1", batch_size=4, max_length=256)

        config2 = DatasetConfig(name="dataset2", batch_size=8, shuffle=True)

        merged = DatasetConfig.merge([config1, config2])

        # Should use config2's batch_size (last one wins)
        assert merged.batch_size == 8
        assert merged.max_length == 256
        assert merged.shuffle == True


class TestHFDatasetWrapper:
    """Test HuggingFace dataset wrapper."""

    @patch("better_ai.data.hf_datasets.load_dataset")
    def test_hf_dataset_loading(self, mock_load_dataset):
        """Test loading HuggingFace dataset."""
        # Mock HF dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(return_value={"text": "sample text"})
        mock_load_dataset.return_value = {"train": mock_dataset}

        wrapper = HFDatasetWrapper(
            dataset_name="test_dataset", split="train", cache_dir="./cache"
        )

        dataset = wrapper.load()
        assert len(dataset) == 100
        mock_load_dataset.assert_called_once()

    @patch("better_ai.data.hf_datasets.load_dataset")
    def test_hf_dataset_preprocessing(self, mock_load_dataset):
        """Test HuggingFace dataset preprocessing."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.__getitem__ = Mock(return_value={"text": "sample text"})
        mock_load_dataset.return_value = {"train": mock_dataset}

        wrapper = HFDatasetWrapper(dataset_name="test_dataset", split="train")

        # Add preprocessing function
        def preprocess_fn(example):
            return {"processed_text": example["text"].upper()}

        wrapper.add_preprocessing(preprocess_fn)
        dataset = wrapper.load()

        # Test preprocessing was applied
        processed_item = dataset[0]
        assert "processed_text" in processed_item

    def test_hf_dataset_streaming(self):
        """Test HuggingFace dataset streaming."""
        wrapper = HFDatasetWrapper(
            dataset_name="test_dataset", split="train", streaming=True
        )

        assert wrapper.streaming == True

        # Mock streaming dataset
        with patch("better_ai.data.hf_datasets.load_dataset") as mock_load:
            mock_streaming_dataset = Mock()
            mock_streaming_dataset.__iter__ = Mock(
                return_value=iter([{"text": "test"}])
            )
            mock_load.return_value = {"train": mock_streaming_dataset}

            dataset = wrapper.load()
            items = list(dataset)
            assert len(items) == 1


class TestDataCollator:
    """Test data collation functionality."""

    def test_basic_collation(self):
        """Test basic data collation."""
        collator = DataCollator(
            tokenizer=Mock(), pad_to_max_length=True, max_length=512
        )

        # Mock batch data
        batch = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5], "attention_mask": [1, 1]},
        ]

        # Mock tokenizer
        collator.tokenizer.pad = Mock(
            return_value={
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            }
        )

        collated = collator(batch)

        assert "input_ids" in collated
        assert "attention_mask" in collated
        assert collated["input_ids"].shape[0] == 2  # batch size

    def test_dynamic_padding(self):
        """Test dynamic padding collation."""
        collator = DataCollator(
            tokenizer=Mock(), padding="longest", return_tensors="pt"
        )

        batch = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6, 7]}]

        # Mock tokenizer for dynamic padding
        collator.tokenizer.pad = Mock(
            return_value={
                "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            }
        )

        collated = collator(batch)

        # Should pad to longest sequence
        assert collated["input_ids"].shape[1] == 4

    def test_custom_collation_fn(self):
        """Test custom collation function."""

        def custom_collate(batch):
            return {
                "custom_input": torch.tensor([item["value"] for item in batch]),
                "metadata": {"batch_size": len(batch)},
            }

        collator = DataCollator(collate_fn=custom_collate)

        batch = [{"value": 1}, {"value": 2}, {"value": 3}]
        collated = collator(batch)

        assert "custom_input" in collated
        assert collated["metadata"]["batch_size"] == 3


class TestTokenizedDataset:
    """Test tokenized dataset functionality."""

    def test_tokenized_dataset_creation(self):
        """Test tokenized dataset creation."""
        # Mock raw dataset
        raw_data = [
            {"text": "First sample"},
            {"text": "Second sample"},
            {"text": "Third sample"},
        ]

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
        }

        dataset = TokenizedDataset(
            raw_data=raw_data,
            tokenizer=mock_tokenizer,
            text_field="text",
            max_length=128,
        )

        assert len(dataset) == 3
        assert dataset.max_length == 128

    def test_tokenization_on_access(self):
        """Test that tokenization happens on access."""
        raw_data = [{"text": "Test text"}]

        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
        }

        dataset = TokenizedDataset(
            raw_data=raw_data, tokenizer=mock_tokenizer, text_field="text"
        )

        # Access should trigger tokenization
        tokenized_item = dataset[0]

        assert "input_ids" in tokenized_item
        assert "attention_mask" in tokenized_item
        mock_tokenizer.assert_called_once_with("Test text")

    def test_caching_tokenization(self):
        """Test tokenization caching."""
        raw_data = [{"text": "Cached text"}]

        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
        }

        dataset = TokenizedDataset(
            raw_data=raw_data,
            tokenizer=mock_tokenizer,
            text_field="text",
            cache_tokenization=True,
        )

        # First access
        item1 = dataset[0]
        # Second access (should use cache)
        item2 = dataset[0]

        # Tokenizer should only be called once
        mock_tokenizer.assert_called_once()
        assert torch.equal(item1["input_ids"], item2["input_ids"])


class TestDataPipeline:
    """Test data pipeline orchestration."""

    def test_pipeline_creation(self):
        """Test data pipeline creation."""
        from better_ai.training.trainer_utils.data import DataPipeline

        # Mock components
        mock_dataset = Mock()
        mock_tokenizer = Mock()
        mock_collator = Mock()

        pipeline = DataPipeline(
            dataset=mock_dataset,
            tokenizer=mock_tokenizer,
            collator=mock_collator,
            batch_size=4,
        )

        assert pipeline.dataset == mock_dataset
        assert pipeline.batch_size == 4

    def test_pipeline_with_transforms(self):
        """Test pipeline with data transforms."""
        from better_ai.training.trainer_utils.data import DataPipeline

        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)

        def transform_fn(example):
            example["transformed"] = True
            return example

        pipeline = DataPipeline(
            dataset=mock_dataset, transforms=[transform_fn], batch_size=2
        )

        # Test transform application
        sample = {"text": "test"}
        transformed = pipeline.apply_transforms(sample)

        assert transformed["transformed"] == True

    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        from better_ai.training.trainer_utils.data import DataPipeline

        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(side_effect=RuntimeError("Data loading error"))

        pipeline = DataPipeline(
            dataset=mock_dataset, batch_size=2, error_handling="skip"
        )

        # Should handle error gracefully
        with pytest.raises(RuntimeError):
            pipeline[0]


class TestDataQualityChecks:
    """Test data quality and validation checks."""

    def test_data_length_validation(self):
        """Test data length validation."""
        from better_ai.training.trainer_utils.data import DataValidator

        validator = DataValidator(max_length=512)

        # Valid data
        valid_data = {"input_ids": [1] * 100}
        assert validator.validate_length(valid_data)

        # Invalid data (too long)
        invalid_data = {"input_ids": [1] * 600}
        assert not validator.validate_length(invalid_data)

    def test_data_content_validation(self):
        """Test data content validation."""
        from better_ai.training.trainer_utils.data import DataValidator

        validator = DataValidator(
            required_fields=["input_ids", "attention_mask"], forbidden_content=None
        )

        # Valid data
        valid_data = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        assert validator.validate_content(valid_data)

        # Invalid data (missing field)
        invalid_data = {"input_ids": [1, 2, 3]}
        assert not validator.validate_content(invalid_data)

    def test_data_statistics_collection(self):
        """Test data statistics collection."""
        from better_ai.training.trainer_utils.data import DataStatistics

        stats = DataStatistics()

        # Collect statistics from batch
        batch = [
            {"input_ids": [1, 2, 3], "length": 3},
            {"input_ids": [4, 5], "length": 2},
            {"input_ids": [6, 7, 8, 9], "length": 4},
        ]

        stats.update(batch)

        assert stats.total_samples == 3
        assert stats.avg_length == 3.0  # (3 + 2 + 4) / 3
        assert stats.max_length == 4
        assert stats.min_length == 2
