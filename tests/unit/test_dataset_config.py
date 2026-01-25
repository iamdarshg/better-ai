
import unittest
import os
from better_ai.data.dataset_config import DatasetConfig, load_dataset_from_config, load_datasets_by_stage

class TestDatasetConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test_datasets.yml'
        with open(self.config_path, 'w') as f:
            f.write("""
datasets:
  - name: "test_dummy"
    type: "dummy"
    path: "path/to/dummy"
    stage: "pretraining"
  - name: "test_another_dummy"
    type: "dummy"
    path: "path/to/another_dummy"
    stage: "sft"
""")

    def tearDown(self):
        os.remove(self.config_path)

    def test_get_dataset_configs(self):
        config = DatasetConfig(self.config_path)
        dataset_configs = config.get_dataset_configs()
        self.assertEqual(len(dataset_configs), 2)
        self.assertEqual(dataset_configs[0]['name'], 'test_dummy')

    def test_load_dataset_from_config(self):
        dataset_config = load_dataset_from_config('test_dummy', self.config_path)
        self.assertEqual(dataset_config['name'], 'test_dummy')
        self.assertEqual(dataset_config['path'], 'path/to/dummy')

    def test_load_nonexistent_dataset(self):
        with self.assertRaises(ValueError):
            load_dataset_from_config('nonexistent', self.config_path)

    def test_load_datasets_by_stage(self):
        pretraining_datasets = load_datasets_by_stage('pretraining', self.config_path)
        self.assertEqual(len(pretraining_datasets), 1)
        self.assertEqual(pretraining_datasets[0]['name'], 'test_dummy')

        sft_datasets = load_datasets_by_stage('sft', self.config_path)
        self.assertEqual(len(sft_datasets), 1)
        self.assertEqual(sft_datasets[0]['name'], 'test_another_dummy')

        rlhf_datasets = load_datasets_by_stage('rlhf', self.config_path)
        self.assertEqual(len(rlhf_datasets), 0)

if __name__ == '__main__':
    unittest.main()
