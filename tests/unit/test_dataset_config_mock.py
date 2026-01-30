#!/usr/bin/env python3
import unittest
from unittest.mock import patch
from better_ai.data.dataset_config import DatasetConfig


class TestDatasetConfigMock(unittest.TestCase):
    def test_get_dataset_configs_with_mocked_safe_load(self):
        import tempfile

        with patch(
            "better_ai.data.dataset_config.yaml.safe_load",
            return_value={"datasets": [{"name": "mocked", "stage": "pretraining"}]},
        ):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yml")
            tmp.write(b"datasets:\n  - name: mocked\n    stage: pretraining\n")
            tmp.close()
            cfg = DatasetConfig(tmp.name)
            datasets = cfg.get_dataset_configs()
            self.assertEqual(len(datasets), 1)
            self.assertEqual(datasets[0]["name"], "mocked")


if __name__ == "__main__":
    unittest.main()
