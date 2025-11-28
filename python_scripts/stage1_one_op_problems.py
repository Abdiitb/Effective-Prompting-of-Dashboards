import random
import json

random.seed(1)

def generate_one_op_samples(n_samples, min_val=1, max_val=999):
    samples = []

    # operator pool with distribution
    operator_pool = (
        ["+"] * 5 +
        ["-"] * 40 +
        ["*"] * 30 +
        ["/"] * 25
    )

    for _ in range(n_samples):
        op = random.choice(operator_pool)

        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)

        # avoid division by zero
        if op == "/":
            b = random.randint(1, max_val)

        # compute answer
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        elif op == "*":
            ans = a * b
        elif op == "/":
            ans = round(a / b, 4)  # control decimals

        # natural language question
        q = f"Solve ({a} {op} {b})"

        # full sentence answer
        a_full = f"The result of ({a} {op} {b}) is {ans}\n\n####{ans}"

        samples.append({
            "question": q,
            "answer": a_full,
            "a": a,
            "b": b,
            "operator": op,
            "result": ans
        })

    return samples


# Example usage
data = generate_one_op_samples(100)

# save as jsonl
with open("stage2_single_op_2_val.jsonl", "w") as f:
    for row in data:
        f.write(json.dumps(row) + "\n")

print("Generated 100 samples.")