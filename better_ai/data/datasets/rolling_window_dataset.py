
import torch
from torch.utils.data import IterableDataset
from typing import List, Dict, Optional, Iterator
from datasets import load_dataset
import gc
import random


class RollingWindowCodeDataset(IterableDataset):
    """Rolling window dataset for streaming code data with minimal memory usage"""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        tokenizer=None,
        max_length: int = 1024,
        languages: Optional[List[str]] = None,
        window_size: int = 1000,
        step_size: int = 500,
        overlap: int = 100,
        max_total_samples: Optional[int] = None,
        cache_dir: str = "./cache",
        use_streaming: bool = True,
        shuffle_buffer_size: int = 1000,
        memory_cleanup_interval: int = 100,
        max_concurrent_samples: int = 5000
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.languages = languages or ["Python", "C", "Rust"]
        self.window_size = window_size
        self.step_size = step_size
        self.overlap = overlap
        self.max_total_samples = max_total_samples
        self.cache_dir = cache_dir
        self.use_streaming = use_streaming
        self.shuffle_buffer_size = shuffle_buffer_size
        self.memory_cleanup_interval = memory_cleanup_interval
        self.max_concurrent_samples = max_concurrent_samples

        print(f"Initializing RollingWindowCodeDataset:")
        print(f"  Dataset: {dataset_name}")
        print(f"  Languages: {self.languages}")
        print(f"  Window size: {window_size}, Step: {step_size}, Overlap: {overlap}")
        print(f"  Streaming: {use_streaming}")

        # Initialize streaming dataset
        try:
            # Skip loading for test datasets
            if dataset_name == 'test':
                self.dataset = None
                print("Using test dataset (synthetic)")
            elif use_streaming:
                self.dataset = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=True,
                    cache_dir=cache_dir
                )
            else:
                self.dataset = load_dataset(
                    dataset_name,
                    split=split,
                    cache_dir=cache_dir
                )
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            self.dataset = None

        # State management
        self.buffer = []
        self.samples_processed = 0
        self.window_count = 0
        self.cleanup_counter = 0

    def _filter_and_process_sample(self, sample: Dict) -> Optional[Dict]:
        """Filter and process a single sample"""
        if not sample:
            return None

        # Get language and content
        language = sample.get('language', sample.get('lang', '')).lower()
        content = sample.get('content', sample.get('text', ''))

        # Map to target languages
        mapped_lang = None
        if 'python' in language:
            mapped_lang = 'Python'
        elif 'c' in language and '++' not in language:
            mapped_lang = 'C'
        elif 'rust' in language:
            mapped_lang = 'Rust'

        if mapped_lang and mapped_lang in self.languages:
            if content and len(content.strip()) > 50:  # Minimum content length
                # Clean content
                cleaned_content = self._clean_code(content)
                if cleaned_content and len(cleaned_content) > 30:
                    return {
                        'content': cleaned_content,
                        'language': mapped_lang,
                        'repo': sample.get('repo_name', 'unknown'),
                        'path': sample.get('path', 'unknown')
                    }

        return None

    def _clean_code(self, content: str) -> str:
        """Clean code content"""
        # Remove excessive blank lines
        lines = content.split('\n')
        cleaned_lines = []
        blank_count = 0

        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(line)
                blank_count = 0
            elif blank_count < 3:  # Allow up to 3 consecutive blank lines
                cleaned_lines.append(line)
                blank_count += 1

        # Limit size
        cleaned_lines = cleaned_lines[:200]  # Max 200 lines
        content = '\n'.join(cleaned_lines)

        # Limit character length
        if len(content) > 4000:  # Max 4K chars
            content = content[:4000] + '\n... [truncated]'

        return content

    def _tokenize_sample(self, sample: Dict) -> Optional[Dict]:
        """Tokenize a processed sample"""
        try:
            content = sample['content']

            # Check if tokenizer is available
            if self.tokenizer is None:
                # Fallback to simple tokenization
                tokens = [hash(c) % 32768 for c in content[:self.max_length]]
            elif hasattr(self.tokenizer, 'encode'):
                # Check if tokenizer supports truncation parameter
                try:
                    tokens = self.tokenizer.encode(content, truncation=True, max_length=self.max_length)
                except TypeError:
                    # Fallback for tokenizers that don't support truncation
                    tokens = self.tokenizer.encode(content, max_length=self.max_length)
                    tokens = tokens[:self.max_length]  # Manual truncation
            else:
                # Fallback to simple tokenization
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
                'language': sample['language'],
                'repo': sample['repo']
            }
        except Exception as e:
            print(f"Tokenization error: {e}")
            return None

    def _cleanup_memory(self):
        """Periodic memory cleanup"""
        self.cleanup_counter += 1
        if self.cleanup_counter >= self.memory_cleanup_interval:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.cleanup_counter = 0

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over rolling windows"""
        if self.dataset is None:
            # Create synthetic data as fallback
            print("Creating synthetic data iterator for training...")
            synthetic_count = 0
            max_samples = self.max_total_samples or 100  # Default limit

            templates = {
                'Python': ['def hello(): print("Hello")'],
                'C': ['int main() { return 0; }'],
                'Rust': ['fn main() { println!("Hello"); }']
            }

            while synthetic_count < max_samples:
                lang = ['Python', 'C', 'Rust'][synthetic_count % 3]
                text = templates[lang][0]

                # Create synthetic sample directly
                if self.tokenizer and hasattr(self.tokenizer, 'encode'):
                    tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
                else:
                    # Fallback tokenization
                    tokens = [hash(c) % 32768 for c in text[:self.max_length]]

                # Pad if needed
                if len(tokens) < self.max_length:
                    padding_length = self.max_length - len(tokens)
                    pad_id = getattr(self.tokenizer, 'pad_token_id', 0) if self.tokenizer else 0
                    tokens = tokens + [pad_id] * padding_length

                yield {
                    'input_ids': torch.tensor(tokens, dtype=torch.long),
                    'attention_mask': torch.ones(self.max_length, dtype=torch.long),
                    'labels': torch.tensor(tokens, dtype=torch.long),
                    'language': lang,
                    'repo': f'synthetic_repo',
                    'path': f'synthetic_file'
                }

                synthetic_count += 1
            return

        # Only reach here if we have a real dataset
        self.buffer = []
        self.samples_processed = 0
        self.window_count = 0

        # Create iterator from dataset
        if self.use_streaming:
            # Try to shuffle, fallback to no shuffle for compatibility
            try:
                dataset_iter = iter(self.dataset.shuffle(seed=42))
            except Exception:
                dataset_iter = iter(self.dataset)
        else:
            dataset_iter = iter(self.dataset)

        dataset_exhausted = False

        while not self.max_total_samples or self.samples_processed < self.max_total_samples:
            if dataset_exhausted and len(self.buffer) == 0:
                # Dataset is exhausted and no more samples in buffer
                print(f"Dataset exhausted after {self.samples_processed} samples")
                break

            # Try to get more samples from dataset
            if not dataset_exhausted:
                try:
                    for _ in range(100):  # Try to get multiple samples at once
                        if self.max_total_samples and self.samples_processed >= self.max_total_samples:
                            break

                        raw_sample = next(dataset_iter)

                        # Process sample
                        processed_sample = self._filter_and_process_sample(raw_sample)
                        if processed_sample:
                            tokenized_sample = self._tokenize_sample(processed_sample)
                            if tokenized_sample:
                                self.buffer.append(tokenized_sample)
                                self.samples_processed += 1

                                # Memory management
                                if len(self.buffer) > self.max_concurrent_samples:
                                    self.buffer = self.buffer[-self.max_concurrent_samples:]

                except StopIteration:
                    dataset_exhausted = True
                    print(f"Dataset iterator exhausted, processed {self.samples_processed} samples")

            # Emit samples if we have enough
            while len(self.buffer) >= self.window_size:
                # Create window
                window_samples = self.buffer[-self.window_size:]
                yield from window_samples

                # Slide the window
                slide_amount = min(self.step_size, len(self.buffer))
                self.buffer = self.buffer[slide_amount:]
                self.window_count += 1

                # Periodic cleanup
                self._cleanup_memory()

                # Progress reporting
                if self.window_count % 10 == 0:
                    print(f"Window {self.window_count}, Processed {self.samples_processed} samples")

        # Emit remaining samples in buffer
        if len(self.buffer) > 0:
            yield from self.buffer

        print(f"Completed {self.window_count} windows, processed {self.samples_processed} total samples")

    def _create_synthetic_iterator(self) -> Iterator[Dict]:
        """Create synthetic data as fallback"""
        print("Creating synthetic code data as fallback...")

        templates = {
            'Python': [
                'def function_{i}():\n    return {val}',
                'class Class_{i}:\n    def method(self):\n        return {val}',
                'import random\ndef generate_data():\n    return [random.randint(1, {val}) for _ in range(10)]'
            ],
            'C': [
                'int function_{i}(int x) {{\n    return x + {val};\n}}',
                'struct struct_{i} {{\n    int field_{j};\n    float value;\n}};',
                '#include <stdio.h>\nint main() {{\n    printf("{val}\\n");\n    return 0;\n}}'
            ],
            'Rust': [
                'fn function_{i}() -> i32 {{\n    {val}\n}}',
                'struct Struct_{i} {{\n    field_{j}: i32,\n    value: f64,\n}}',
                'fn main() {{\n    let result = {val};\n    println!("{{}}", result);\n}}'
            ]
        }

        synthetic_count = 0
        max_samples = self.max_total_samples or 1000  # Default to 1000 for safety
        while synthetic_count < max_samples:
            lang = random.choice(list(templates.keys()))
            template = random.choice(templates[lang])

            text = template.format(
                i=random.randint(1, 1000),
                j=random.randint(1, 100),
                val=random.randint(1, 10000)
            )

            # Create synthetic sample
            tokenized_sample = self._tokenize_sample({
                'content': text,
                'language': lang,
                'repo': f'synthetic_repo',
                'path': f'synthetic_file'
            })

            if tokenized_sample:
                yield tokenized_sample
                synthetic_count += 1
