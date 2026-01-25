
import unittest
import os
from better_ai.data.dataset_config import DatasetConfig, load_dataset_from_config

class TestDatasetConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test_datasets.yml'
        with open(self.config_path, 'w') as f:
            f.write("""
datasets:
  - name: "test_dummy"
    type: "dummy"
    path: "path/to/dummy"
  - name: "test_another_dummy"
    type: "dummy"
    path: "path/to/another_dummy"
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

if __name__ == '__main__':
    unittest.main()
