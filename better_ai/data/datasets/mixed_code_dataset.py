
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from datasets import load_dataset


class MixedCodeDataset(Dataset):
    """Mixed dataset combining multiple programming languages"""

    def __init__(
        self,
        dataset_configs: List[Dict],
        tokenizer=None,
        max_length: int = 1024,
        total_max_samples: Optional[int] = None,
        cache_dir: str = "./cache"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir

        print("Loading mixed code datasets...")

        all_data = []

        for config in dataset_configs:
            dataset_name = config['name']
            languages = config['languages']
            max_samples = config.get('max_samples', None)

            print(f"\nLoading {dataset_name} with languages: {languages}")

            try:
                raw_dataset = load_dataset(dataset_name, cache_dir=cache_dir)

                # Process dataset
                filtered_data = self._process_dataset(
                    raw_dataset, languages, max_samples
                )

                all_data.extend(filtered_data)
                print(f"  Added {len(filtered_data)} samples")

            except Exception as e:
                print(f"  Failed to load {dataset_name}: {e}")
                continue

        # Shuffle and limit total samples
        import random
        random.shuffle(all_data)

        if total_max_samples:
            all_data = all_data[:total_max_samples]

        self.data = all_data
        print(f"\nTotal mixed dataset size: {len(self.data)} samples")

        # Show distribution
        lang_counts = {}
        for item in self.data:
            lang = item['language']
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        print("Final language distribution:")
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count} ({count/len(self.data)*100:.1f}%)")

    def _process_dataset(self, raw_dataset, target_languages: List[str], max_samples: Optional[int]):
        """Process raw dataset based on target languages"""
        processed = []
        count = 0

        for example in raw_dataset:
            if max_samples and count >= max_samples:
                break

            content = example.get('content', example.get('text', ''))
            language = example.get('language', '').lower()

            # Map to target languages
            mapped_lang = None
            if 'python' in language:
                mapped_lang = 'Python'
            elif 'c' in language and '++' not in language:
                mapped_lang = 'C'
            elif 'rust' in language:
                mapped_lang = 'Rust'

            if mapped_lang and mapped_lang in target_languages:
                if content and len(content.strip()) > 0:
                    # Clean content
                    cleaned_content = content[:8000]  # Limit size

                    processed.append({
                        'content': cleaned_content,
                        'language': mapped_lang,
                        'repo': example.get('repo_name', 'unknown'),
                        'path': example.get('path', 'unknown')
                    })
                    count += 1

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        content = example['content']

        # Tokenize
        if self.tokenizer and hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(content, truncation=True, max_length=self.max_length)
        else:
            # Fallback tokenization
            tokens = [hash(c) % 32768 for c in content[:self.max_length]]

        # Pad if needed
        if len(tokens) < self.max_length:
            padding_length = self.max_length - len(tokens)
            pad_id = getattr(self.tokenizer, 'pad_token_id', 0) if self.tokenizer else 0
            tokens = tokens + [pad_id] * padding_length

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.ones(self.max_length, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long),
            'language': example['language'],
            'repo': example['repo']
        }
