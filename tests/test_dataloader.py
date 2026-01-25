import unittest
from unittest.mock import Mock, patch
from better_ai.data.unified_dataloader import StreamingDataset

class TestDataloader(unittest.TestCase):
    @patch('better_ai.data.unified_dataloader.load_dataset')
    def test_xml_formatting(self, mock_load_dataset):
        mock_load_dataset.return_value = [
            {
                "problem": "This is a problem.",
                "constraints": "These are the constraints.",
                "examples": "These are the examples.",
            }
        ]

        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock(),
        }

        dataset = StreamingDataset(
            dataset_name="dummy",
            tokenizer=tokenizer,
            max_length=128,
        )

        item = {
            "problem": "This is a problem.",
            "constraints": "These are the constraints.",
            "examples": "These are the examples.",
        }

        formatted_text = dataset._format_with_xml(item)

        self.assertEqual(
            formatted_text,
            "<problem>This is a problem.</problem><constraints>These are the constraints.</constraints><examples>These are the examples.</examples>"
        )

if __name__ == "__main__":
    unittest.main()
