import pytest
import os
import yaml
import json
from better_ai.config import TrainingConfig

def test_training_config_to_from_file_json(tmp_path):
    config = TrainingConfig(
        batch_size=4,
        learning_rate=2e-5,
        max_steps=1000
    )

    json_path = tmp_path / "config.json"
    config.to_file(str(json_path))

    assert os.path.exists(json_path)

    loaded_config = TrainingConfig.from_file(str(json_path))
    assert loaded_config.batch_size == 4
    assert loaded_config.learning_rate == 2e-5
    assert loaded_config.max_steps == 1000
    assert loaded_config.optimizer == "adamw" # default

def test_training_config_to_from_file_yaml(tmp_path):
    config = TrainingConfig(
        batch_size=8,
        learning_rate=1e-4,
        max_steps=500
    )

    yaml_path = tmp_path / "config.yaml"
    config.to_file(str(yaml_path))

    assert os.path.exists(yaml_path)

    loaded_config = TrainingConfig.from_file(str(yaml_path))
    assert loaded_config.batch_size == 8
    assert loaded_config.learning_rate == 1e-4
    assert loaded_config.max_steps == 500

def test_training_config_from_yaml_content(tmp_path):
    yaml_content = """
batch_size: 16
learning_rate: 5.0e-5
max_steps: 2000
"""
    yaml_path = tmp_path / "manual_config.yml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    loaded_config = TrainingConfig.from_file(str(yaml_path))
    assert loaded_config.batch_size == 16
    assert loaded_config.learning_rate == 5e-5
    assert loaded_config.max_steps == 2000
