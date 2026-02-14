
import json
import random

def generate_seed_data(num_samples=100):
    """
    Generates high-quality seed data with varying length distributions.
    (7.15 Seed Data)
    """
    domains = ["code", "math", "reasoning", "general"]
    lengths = [
        ("short", 10, 50),
        ("medium", 50, 200),
        ("long", 200, 500),
        ("very_long", 500, 1000)
    ]

    seed_data = []
    for i in range(num_samples):
        domain = random.choice(domains)
        length_name, min_len, max_len = random.choice(lengths)

        # Mock content generation
        content = f"This is a {length_name} response for a {domain} task. " * (min_len // 5)

        seed_data.append({
            "instruction": f"Instruction {i} for {domain}",
            "response": content,
            "domain": domain,
            "length_category": length_name
        })

    return seed_data

if __name__ == "__main__":
    data = generate_seed_data()
    with open("data/seed_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} seed samples")
