import random
import json

def generate_two_op_samples(
    n_samples,
    min_val=1,
    max_val=200,
    float_prob=0.3,
    float_precision=2
):
    samples = []

    # operator distribution for two-operations
    # + : 30%, - : 30%, * : 25%, / : 15%
    operators = ["+"] * 30 + ["-"] * 30 + ["*"] * 25 + ["/"] * 15

    for _ in range(n_samples):
        op1 = random.choice(operators)
        op2 = random.choice(operators)

        # decide if operands are float or int
        def random_number():
            if random.random() < float_prob:
                return round(random.uniform(min_val, max_val), float_precision)
            else:
                return random.randint(min_val, max_val)

        a, b, c = random_number(), random_number(), random_number()

        # avoid division by zero
        if op1 == "/" and b == 0:
            b = random.uniform(1, max_val)
        if op2 == "/" and c == 0:
            c = random.uniform(1, max_val)

        # create expression string
        expr = f"({a} {op1} {b}) {op2} {c}"

        # safely compute numeric result
        try:
            result = eval(expr)
            if isinstance(result, float):
                result = round(result, float_precision)
        except ZeroDivisionError:
            continue  # skip invalid division

        # question and full-sentence answer
        question = f"Solve: {expr}"
        answer = f"The result of {expr} is {result}"

        samples.append({
            "question": question,
            "answer": answer,
            "a": a,
            "b": b,
            "c": c,
            "op1": op1,
            "op2": op2,
            "result": result
        })

    return samples


# Example usage
data = generate_two_op_samples(5000)

# Save to JSONL
with open("stage1_two_op.jsonl", "w") as f:
    for row in data:
        f.write(json.dumps(row) + "\n")

print("Generated 10 two-operation arithmetic samples with full-sentence answers.")