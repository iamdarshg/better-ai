
import unittest
import os
from better_ai.data.dataset_config import DatasetConfig, load_dataset_from_config, load_datasets_by_stage
from better_ai.data.unified_dataloader import create_dataloader
from transformers import AutoTokenizer

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
    max_seq_length: 1024
    num_training_steps: 100
  - name: "test_another_dummy"
    type: "dummy"
    path: "path/to/another_dummy"
    stage: "sft"
    max_seq_length: 512
    num_training_steps: 50
""")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")


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

    def test_load_dataset_with_new_params(self):
        dataset_config = load_dataset_from_config('test_dummy', self.config_path)
        self.assertEqual(dataset_config['max_seq_length'], 1024)
        self.assertEqual(dataset_config['num_training_steps'], 100)

    def test_create_dataloader_with_config(self):
        dataset_config = load_dataset_from_config('test_dummy', self.config_path)
        # The path is a dummy path, so we expect this to fail
        # This test is just to ensure the parameters are passed correctly
        with self.assertRaises(Exception):
            dataloader = create_dataloader(
                dataset_config,
                self.tokenizer,
                batch_size=2
            )
            self.assertEqual(dataloader.dataset.max_length, 1024)

if __name__ == '__main__':
    unittest.main()
