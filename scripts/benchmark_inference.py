import torch
import time
import argparse
from tqdm import tqdm
from better_ai.config import ModelConfig
from better_ai.models.core import DeepSeekModel
from transformers import AutoTokenizer

def benchmark_inference(model, tokenizer, prompt, num_tokens=50, num_iters=10):
    """
    Benchmarks token generation speed.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device}...")
    model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Warmup
    print("Warming up...")
    for _ in range(2):
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=5)

    latencies = []
    throughputs = []

    for _ in tqdm(range(num_iters), desc="Benchmarking"):
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=num_tokens)
        end_time = time.time()

        duration = end_time - start_time
        num_generated = output.shape[1] - input_ids.shape[1]

        if num_generated > 0:
            latencies.append(duration / num_generated)
            throughputs.append(num_generated / duration)

    if not latencies:
        print("Error: No tokens generated during benchmark.")
        return

    avg_latency = sum(latencies) / len(latencies)
    avg_throughput = sum(throughputs) / len(throughputs)

    print(f"\nResults:")
    print(f"Average Latency: {avg_latency*1000:.2f} ms/token")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/sec")

    if torch.cuda.is_available():
        print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Write a quicksort algorithm in Python.")
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--small", action="store_true", help="Use small model for testing")
    args = parser.parse_args()

    if args.small:
        config = ModelConfig.get_small_model_config()
        model = DeepSeekModel(config)
    else:
        # In a real environment, we'd load a checkpoint
        config = ModelConfig()
        model = DeepSeekModel(config)

    # Using a fast tokenizer for benchmarking
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")

    benchmark_inference(model, tokenizer, args.prompt, num_tokens=args.tokens, num_iters=args.iters)
