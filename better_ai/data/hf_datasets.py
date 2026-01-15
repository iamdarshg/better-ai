"""Focused Hugging Face datasets for agentic coding (Python, C, Rust)"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Dict, Any, Optional, Iterator
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
import gc
import random


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


class ExpertAwareRollingDataset(RollingWindowCodeDataset):
    """Rolling window dataset with expert-aware data routing for MoE training"""
    
    def __init__(
        self,
        num_experts: int = 8,
        experts_per_token: int = 2,
        expert_capacity_factor: float = 1.25,
        expert_buffer_size: int = 200,
        load_balance_weight: float = 0.01,
        batch_size: int = 32,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.expert_capacity_factor = expert_capacity_factor
        self.expert_buffer_size = expert_buffer_size
        self.load_balance_weight = load_balance_weight
        self.batch_size = batch_size
        
        # Expert-specific buffers
        self.expert_buffers = {i: [] for i in range(num_experts)}
        self.expert_usage = {i: 0 for i in range(num_experts)}
        
        print(f"ExpertAwareRollingDataset initialized:")
        print(f"  Experts: {num_experts}, Active per token: {experts_per_token}")
        print(f"  Expert buffer size: {expert_buffer_size}")
        print(f"  Batch size: {batch_size}")
    
    def _route_sample_to_experts(self, sample: Dict) -> List[int]:
        """Simple routing based on language hash (in real implementation, use actual gating)"""
        content = sample.get('content', '')
        language = sample.get('language', '')
        
        # Simple hash-based routing for demonstration
        content_hash = hash(content) % self.num_experts
        lang_hash = hash(language) % self.num_experts
        
        # Return top-k experts
        scores = [(i, abs(i - content_hash) + abs(i - lang_hash)) for i in range(self.num_experts)]
        scores.sort(key=lambda x: x[1])
        
        return [expert_id for expert_id, _ in scores[:self.experts_per_token]]
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate with expert-aware routing"""
        if self.dataset is None:
            # Use parent synthetic fallback
            for sample in self._create_synthetic_iterator():
                yield sample
            return
        
        # Reset state
        self.expert_buffers = {i: [] for i in range(self.num_experts)}
        self.samples_processed = 0
        self.window_count = 0
        
        # Create iterator
        if self.use_streaming:
            # Try to shuffle, fallback to no shuffle for compatibility
            try:
                dataset_iter = iter(self.dataset.shuffle(seed=42))
            except Exception:
                dataset_iter = iter(self.dataset)
        
        for raw_sample in dataset_iter:
            if self.max_total_samples and self.samples_processed >= self.max_total_samples:
                break
            
            # Process sample
            processed_sample = self._filter_and_process_sample(raw_sample)
            if processed_sample:
                tokenized_sample = self._tokenize_sample(processed_sample)
                if tokenized_sample:
                    # Route to experts
                    expert_ids = self._route_sample_to_experts(processed_sample)
                    
                    # Add to expert buffers
                    for expert_id in expert_ids:
                        expert_buffer = self.expert_buffers[expert_id]
                        expert_buffer.append(tokenized_sample)
                        self.expert_usage[expert_id] += 1
                        
                        # Maintain buffer size
                        if len(expert_buffer) > self.expert_buffer_size:
                            expert_buffer = expert_buffer[-self.expert_buffer_size:]
                        self.expert_buffers[expert_id] = expert_buffer
                    
                    self.samples_processed += 1
            
            # Emit batches when experts have enough data
            min_buffer_size = min(len(buf) for buf in self.expert_buffers.values())
            if min_buffer_size >= self.batch_size if hasattr(self, 'batch_size') else 32:
                # Create balanced batch across experts
                batch = []
                for expert_id, buffer in self.expert_buffers.items():
                    samples_to_take = min(len(buffer), self.batch_size // self.num_experts if hasattr(self, 'batch_size') else 4)
                    batch.extend(buffer[-samples_to_take:])
                
                # Shuffle batch
                random.shuffle(batch)
                
                # Emit samples
                for sample in batch:
                    yield sample
                
                # Clear used samples from buffers
                for expert_id in self.expert_buffers:
                    samples_to_remove = min(len(self.expert_buffers[expert_id]), self.batch_size // self.num_experts if hasattr(self, 'batch_size') else 4)
                    self.expert_buffers[expert_id] = self.expert_buffers[expert_id][samples_to_remove:]
                
                self.window_count += 1
                
                # Cleanup
                self._cleanup_memory()
                
                # Progress
                if self.window_count % 5 == 0:
                    print(f"Expert window {self.window_count}, Samples: {self.samples_processed}")
                    print(f"Expert usage: {self.expert_usage}")
        
        # Emit remaining samples
        remaining_samples = []
        for buffer in self.expert_buffers.values():
            remaining_samples.extend(buffer)
        
        random.shuffle(remaining_samples)
        for sample in remaining_samples:
            yield sample


def custom_collate_fn(batch):
    """Handle batch collation with mixed data types for MoE training"""
    if len(batch) == 0:
        return {}
    
    # Handle case where batch items might be lists or tensors
    collated = {}
    
    # Standard tensor fields
    tensor_fields = ['input_ids', 'attention_mask', 'labels']
    for field in tensor_fields:
        if field in batch[0]:
            values = []
            for item in batch:
                value = item[field]
                if isinstance(value, list):
                    value = torch.tensor(value, dtype=torch.long)
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.long)
                values.append(value)
            collated[field] = torch.stack(values)
    
    # String fields
    string_fields = ['language', 'repo']
    for field in string_fields:
        if field in batch[0]:
            collated[field] = [item[field] for item in batch]
    
    return collated


def create_code_dataloaders(config, tokenizer, batch_size=4, num_workers=0):
    """Create dataloaders for code datasets with rolling window support"""
    
    # Rolling window configuration
    use_rolling_windows = config.get('use_rolling_windows', True)
    use_streaming = config.get('use_streaming', True)
    expert_aware = config.get('expert_aware', False)
    
    if not use_rolling_windows:
        # Use original implementation
        return _create_legacy_dataloaders(config, tokenizer, batch_size, num_workers)
    
    print(f"Creating rolling window dataloaders:")
    print(f"  Streaming: {use_streaming}")
    print(f"  Expert-aware: {expert_aware}")
    
    # Rolling window parameters
    window_size = config.get('rolling_window_size', 1000)
    step_size = config.get('rolling_step_size', 500)
    overlap = config.get('rolling_overlap', 100)
    max_concurrent = config.get('max_concurrent_samples', 5000)
    memory_cleanup_interval = config.get('memory_cleanup_interval', 100)
    
    # Languages
    languages = config.get('languages', ['Python', 'C', 'Rust'])
    
    # Training dataset
    train_dataset_kwargs = {
        'dataset_name': config.get('primary_dataset', 'bigcode/the-stack'),
        'split': 'train',
        'tokenizer': tokenizer,
        'max_length': config.get('max_length', 1024),
        'languages': languages,
        'window_size': window_size,
        'step_size': step_size,
        'overlap': overlap,
        'max_total_samples': config.get('max_train_samples', 100000),
        'cache_dir': config.get('cache_dir', './cache'),
        'use_streaming': use_streaming,
        'shuffle_buffer_size': config.get('shuffle_buffer_size', 1000),
        'memory_cleanup_interval': memory_cleanup_interval,
        'max_concurrent_samples': max_concurrent
    }
    
    if expert_aware:
        train_dataset = ExpertAwareRollingDataset(
            num_experts=config.get('num_experts', 8),
            experts_per_token=config.get('experts_per_token', 2),
            expert_buffer_size=config.get('expert_buffer_size', 200),
            batch_size=batch_size,
            **train_dataset_kwargs
        )
    else:
        train_dataset = RollingWindowCodeDataset(**train_dataset_kwargs)
    
    # Evaluation dataset (smaller parameters)
    eval_kwargs = train_dataset_kwargs.copy()
    eval_kwargs.update({
        'split': 'train',  # bigcode/the-stack only has train split
        'window_size': min(window_size, 500),
        'step_size': min(step_size, 250),
        'max_total_samples': config.get('max_eval_samples', 10000),
        'max_concurrent_samples': min(max_concurrent, 2000)
    })
    
    if expert_aware:
        eval_dataset = ExpertAwareRollingDataset(
            num_experts=config.get('num_experts', 8),
            experts_per_token=config.get('experts_per_token', 2),
            expert_buffer_size=config.get('expert_buffer_size', 100),
            batch_size=batch_size,
            **eval_kwargs
        )
    else:
        eval_dataset = RollingWindowCodeDataset(**eval_kwargs)
    

    
    # Create dataloaders with appropriate settings
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset doesn't support shuffle
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_dataloader, eval_dataloader


def _create_legacy_dataloaders(config, tokenizer, batch_size=4, num_workers=0):
    """Legacy dataloader creation (original implementation)"""
    print("Using legacy dataloader implementation (non-streaming)")
    
    # Dataset configurations for Python, C, Rust
    dataset_configs = [
        {
            'name': 'bigcode/the-stack',
            'languages': ['Python'],
            'max_samples': config.get('max_python_samples', 500000)
        },
        {
            'name': 'bigcode/the-stack',
            'languages': ['C'],
            'max_samples': config.get('max_c_samples', 300000)
        },
        {
            'name': 'bigcode/the-stack',
            'languages': ['Rust'],
            'max_samples': config.get('max_rust_samples', 200000)
        }
    ]
    
    # Create training dataset
    train_dataset = MixedCodeDataset(
        dataset_configs=dataset_configs,
        tokenizer=tokenizer,
        max_length=config.get('max_length', 1024),
        total_max_samples=config.get('max_train_samples', 100000),
        cache_dir=config.get('cache_dir', './cache')
    )
    
    # Create evaluation dataset (smaller)
    eval_dataset = MixedCodeDataset(
        dataset_configs=[
            {**dc, 'max_samples': min(dc.get('max_samples', 10000), 5000)} 
            for dc in dataset_configs
        ],
        tokenizer=tokenizer,
        max_length=config.get('max_length', 1024),
        total_max_samples=config.get('max_eval_samples', 10000),
        cache_dir=config.get('cache_dir', './cache')
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_dataloader, eval_dataloader
