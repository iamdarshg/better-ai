"""Focused Hugging Face datasets for agentic coding (Python, C, Rust)"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from datasets import load_dataset


class CodeDataset(Dataset):
    """Code dataset for specific languages (Python, C, Rust)"""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        tokenizer=None,
        max_length: int = 1024,
        max_samples: Optional[int] = None,
        language_filter: Optional[List[str]] = None,
        cache_dir: str = "./cache"
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language_filter = language_filter or ["Python", "C", "Rust"]
        self.cache_dir = cache_dir

        print(f"Loading code dataset: {dataset_name} - {split}")
        print(f"Languages: {self.language_filter}")

        # Load dataset
        try:
            raw_dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir
            )
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            # Fallback to synthetic data
            self.data = self._create_synthetic_data(max_samples or 1000)
            return

        # Filter by language
        filtered_data = []
        for example in raw_dataset:
            language = example.get('language', '').lower()

            # Map to our target languages
            if any(lang in language.lower() for lang in ['python', 'c', 'rust']):
                # Normalize language name
                if 'python' in language:
                    normalized_lang = 'Python'
                elif 'c' in language and '++' not in language:
                    normalized_lang = 'C'
                elif 'rust' in language:
                    normalized_lang = 'Rust'
                else:
                    continue

                if normalized_lang in self.language_filter:
                    # Get content
                    content = example.get('content', example.get('text', ''))
                    if content and len(content.strip()) > 0:
                        # Clean content
                        cleaned_content = self._clean_code(content)
                        if cleaned_content:
                            filtered_data.append({
                                'content': cleaned_content,
                                'language': normalized_lang,
                                'repo': example.get('repo_name', 'unknown'),
                                'path': example.get('path', 'unknown')
                            })

        # Limit samples
        if max_samples:
            filtered_data = filtered_data[:max_samples]

        self.data = filtered_data
        print(f"Loaded {len(self.data)} code samples")

        # Show distribution
        lang_counts = {}
        for item in self.data:
            lang = item['language']
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        print("Language distribution:")
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count}")

    def _clean_code(self, content: str) -> str:
        """Clean code content"""
        # Remove excessive blank lines
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep line if it has content or it's the first line
            if stripped or not cleaned_lines:
                cleaned_lines.append(line)
            # Keep some blank lines but not too many
            elif len(cleaned_lines) - len([l for l in cleaned_lines[-5:] if l.strip()]) < 3:
                cleaned_lines.append(line)

        # Limit to reasonable size
        cleaned_lines = cleaned_lines[:500]  # Max 500 lines

        # Limit character length
        content = '\n'.join(cleaned_lines)
        if len(content) > 10000:  # Max 10K chars
            content = content[:10000] + '\n... [truncated]'

        return content

    def _create_synthetic_data(self, num_samples: int) -> List[Dict]:
        """Create synthetic code data as fallback"""
        print(f"Creating {num_samples} synthetic code samples...")

        # Language templates
        templates = {
            'Python': [
                'def hello_world():\n    print("Hello, World!")\n\nhello_world()',
                'class Calculator:\n    def add(self, a, b):\n        return a + b\n\ncalc = Calculator()\nprint(calc.add(5, 3))',
                'import math\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\nprint(factorial(5))'
            ],
            'C': [
                '#include <stdio.h>\n\nint main() {\n    printf("Hello, World!\\n");\n    return 0;\n}',
                'int add(int a, int b) {\n    return a + b;\n}\n\nint main() {\n    printf("%d\\n", add(5, 3));\n    return 0;\n}',
                '#include <stdlib.h>\n\nint factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n-1);\n}\n\nint main() {\n    printf("%d\\n", factorial(5));\n    return 0;\n}'
            ],
            'Rust': [
                'fn main() {\n    println!("Hello, World!");\n}',
                'fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\nfn main() {\n    println!("{}", add(5, 3));\n}',
                'fn factorial(n: u32) -> u32 {\n    if n <= 1 { 1 } else { n * factorial(n - 1) }\n}\n\nfn main() {\n    println!("{}", factorial(5));\n}'
            ]
        }

        data = []
        for i in range(num_samples):
            lang = ['Python', 'C', 'Rust'][i % 3]
            template = templates[lang][i % len(templates[lang])]

            data.append({
                'content': template,
                'language': lang,
                'repo': f'synthetic_repo_{i}',
                'path': f'synthetic_file_{i}.{"py" if lang == "Python" else "c" if lang == "C" else "rs"}'
            })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        content = example['content']

        # Tokenize
        tokens = self.tokenizer.encode(content, truncation=True, max_length=self.max_length)

        # Pad if needed
        if len(tokens) < self.max_length:
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * padding_length

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.ones(self.max_length, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long),
            'language': example['language'],
            'repo': example['repo']
        }
