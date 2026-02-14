import torch
import time
import argparse
from tqdm import tqdm

def benchmark_inference(model, tokenizer, prompt, num_tokens=50, num_iters=10):
    """
    Benchmarks token generation speed.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device}...")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Warmup
    for _ in range(2):
        _ = model.generate(input_ids, max_new_tokens=5)

    latencies = []
    throughputs = []

    for _ in tqdm(range(num_iters), desc="Benchmarking"):
        start_time = time.time()
        output = model.generate(input_ids, max_new_tokens=num_tokens)
        end_time = time.time()

        duration = end_time - start_time
        num_generated = output.shape[1] - input_ids.shape[1]

        latencies.append(duration / num_generated)
        throughputs.append(num_generated / duration)

    avg_latency = sum(latencies) / len(latencies)
    avg_throughput = sum(throughputs) / len(throughputs)

    print(f"\nResults:")
    print(f"Average Latency: {avg_latency*1000:.2f} ms/token")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/sec")

    if torch.cuda.is_available():
        print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    # Mocking for standalone script run
    class MockTokenizer:
        def encode(self, text, **kwargs): return torch.zeros((1, 10), dtype=torch.long)
    class MockModel:
        def generate(self, input_ids, max_new_tokens=50, **kwargs):
            return torch.zeros((1, input_ids.shape[1] + max_new_tokens))

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Write a quicksort algorithm in Python.")
    parser.add_argument("--tokens", type=int, default=50)
    args = parser.parse_args()

    benchmark_inference(MockModel(), MockTokenizer(), args.prompt, num_tokens=args.tokens)
