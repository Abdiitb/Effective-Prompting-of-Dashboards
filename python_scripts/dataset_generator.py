# Generating a 25,000-sample arithmetic dataset (depth 6-7) with Chain-of-Thought (CoT) step-by-step reasoning.
# Output: JSONL file at /mnt/data/arithmetic_cot_25k.jsonl with entries containing "system","user","assistant".
# Deterministic seed for reproducibility.
# Features:
# - Operators: +, -, *, /
# - Parentheses inserted by expression tree structure
# - Mix of integers and floats
# - Avoids division by zero
# - CoT: step-by-step post-order evaluation steps followed by "Final answer: <value>"
# - Configurable sample_count, depth_range, float probability, value ranges
# - Lightweight evaluator using Python floats (rounded to 6 significant digits where appropriate)

import random
import json
import math
from typing import Tuple, List

random.seed(42)

OUTPUT_PATH = "/mnt/data/arithmetic_cot_25k.jsonl"
SAMPLE_COUNT = 25000
DEPTH_MIN = 6
DEPTH_MAX = 7
OPERATORS = ['+', '-', '*', '/']
PROB_FLOAT = 0.4  # fraction of numbers that are floats
INT_RANGE = (0, 200)   # typical integer magnitude
FLOAT_RANGE = (0.0, 200.0)  # typical float magnitude

SYSTEM_TEXT = (
    "You are a math solver. "
    "Solve arithmetic expressions. Provide step-by-step reasoning, then the final numeric answer."
)

# Helper to format numbers neatly
def fmt_number(x: float) -> str:
    # If it's nearly an integer, print as integer
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    # Otherwise format with up to 6 decimal places, trim trailing zeros
    s = f"{x:.6f}".rstrip('0').rstrip('.')
    # Avoid '-0'
    if s == '-0':
        s = '0'
    return s

# Node classes for expression tree
class Node:
    pass

class Num(Node):
    def __init__(self, value: float, raw_repr: str):
        self.value = value
        self.raw = raw_repr  # textual representation for expression string

class Op(Node):
    def __init__(self, left: Node, op: str, right: Node):
        self.left = left
        self.op = op
        self.right = right

# Generate a random number node (int or float)
def gen_number_node() -> Num:
    if random.random() < PROB_FLOAT:
        v = random.uniform(FLOAT_RANGE[0], FLOAT_RANGE[1])
        # round a bit to avoid super long decimals
        v = round(v, 6)
        # avoid exactly 0 for divisors situations - we'll handle in evaluator
        if abs(v) < 1e-6:
            v = 1.0
        repr_str = fmt_number(v)
        return Num(v, repr_str)
    else:
        v = random.randint(INT_RANGE[0], INT_RANGE[1])
        repr_str = str(v)
        return Num(float(v), repr_str)

# Build an expression tree with target operations count = depth (number of ops)
def build_expr_tree(n_ops: int) -> Node:
    # We need n_ops operators, hence n_ops+1 numbers. We'll build by repeatedly combining.
    nodes: List[Node] = [gen_number_node() for _ in range(n_ops + 1)]
    ops = [random.choice(OPERATORS) for _ in range(n_ops)]
    # Randomly combine adjacent nodes, inserting parentheses via tree structure
    idxs = list(range(len(nodes)))
    while len(nodes) > 1:
        # choose an index to combine - weighted to encourage nesting
        i = random.randrange(len(nodes) - 1)
        left = nodes[i]
        right = nodes[i+1]
        op = ops.pop(0)
        new_node = Op(left, op, right)
        # replace nodes[i:i+2] with new_node
        nodes[i:i+2] = [new_node]
    return nodes[0]

# Convert tree to string with parentheses where needed
def tree_to_string(node: Node) -> str:
    if isinstance(node, Num):
        return node.raw
    assert isinstance(node, Op)
    left_s = tree_to_string(node.left)
    right_s = tree_to_string(node.right)
    return f"({left_s} {node.op} {right_s})"

# Evaluate tree with post-order traversal and record steps
def eval_with_steps(node: Node) -> Tuple[float, List[str]]:
    if isinstance(node, Num):
        return node.value, [node.raw]
    assert isinstance(node, Op)
    left_val, left_steps = eval_with_steps(node.left)
    right_val, right_steps = eval_with_steps(node.right)

    # Avoid division by zero by adjusting right_val slightly if zero (rare)
    if node.op == '/' and abs(right_val) < 1e-12:
        # replace right_val with 1.0, and record the adjustment
        right_val = 1.0
        right_steps = right_steps + ["(Adjusted denominator from 0 to 1 to avoid division by zero)"]

    # Compute result
    if node.op == '+':
        res = left_val + right_val
    elif node.op == '-':
        res = left_val - right_val
    elif node.op == '*':
        res = left_val * right_val
    elif node.op == '/':
        res = left_val / right_val
    else:
        raise ValueError("Unknown op")

    # Format operands for step string: prefer formatted numbers
    left_str = fmt_number(left_val)
    right_str = fmt_number(right_val)
    res_str = fmt_number(res)

    step = f"{left_str} {node.op} {right_str} = {res_str}"
    # Combine steps: left steps (excluding raw numbers to avoid clutter), then right steps (if any), then this op
    combined_steps = []
    # To keep steps readable, include only meaningful computation lines.
    # left_steps and right_steps may include raw numbers from leaves; we'll include them
    combined_steps.extend([s for s in left_steps if isinstance(s, str)])
    combined_steps.extend([s for s in right_steps if isinstance(s, str)])
    combined_steps.append(step)
    return res, combined_steps

# Compute a single sample; returns dict to write as JSONL line
def make_sample(max_depth_range=(6,7)) -> dict:
    # Select depth (number of ops) uniformly between min and max
    n_ops = random.randint(max_depth_range[0], max_depth_range[1])
    tree = build_expr_tree(n_ops)
    expr = tree_to_string(tree)
    # Clean expression: remove outermost parentheses if desired
    if expr.startswith("(") and expr.endswith(")"):
        expr_clean = expr[1:-1]
    else:
        expr_clean = expr
    # Build user prompt
    user_text = f"Solve: {expr_clean}. Show steps."
    # Evaluate and collect steps
    try:
        value, steps = eval_with_steps(tree)
    except Exception as e:
        # As a fallback, evaluate with python's eval safely
        safe_expr = expr_clean.replace('^', '**')
        value = float(eval(safe_expr))
        steps = [f"Evaluated {expr_clean} = {fmt_number(value)}"]
    assistant_text = "\n".join(steps) + f"\nFinal answer: {fmt_number(value)}"
    return {"system": SYSTEM_TEXT, "user": user_text, "assistant": assistant_text}

# Generate dataset and write JSONL
def generate_dataset(path: str, count: int, depth_min: int, depth_max: int):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            if (i+1) % 1000 == 0:
                print(f"Generating sample {i+1}/{count}")
            sample = make_sample((depth_min, depth_max))
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# Run generation
generate_dataset(OUTPUT_PATH, SAMPLE_COUNT, DEPTH_MIN, DEPTH_MAX)

print(f"Dataset generation complete. Wrote {SAMPLE_COUNT} samples to {OUTPUT_PATH}")
