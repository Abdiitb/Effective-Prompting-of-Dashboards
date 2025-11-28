import random
import json

random.seed(0)

ONE_OP_SAMPLES = 10000
OUTPUT_FILE = "stage1_mixed_one_two_ops_200.jsonl"

ops = ["+", "-", "*", "/"]

def random_number():
    if random.random() < 0.5:
        return str(random.randint(1, 50))
    else:
        return str(round(random.uniform(1, 50), 2))

def safe_eval(expr):
    try:
        return round(eval(expr), 4)
    except Exception:
        return None

# -------------------------------
# ONE-OPERATION SAMPLE
# -------------------------------
def gen_one_op():
    A = random_number()
    B = random_number()
    op = random.choice(ops)
    expr = f"( {A} {op} {B} )"

    val = safe_eval(expr)
    if val is None:
        return gen_one_op()

    return {
        "user": f"Solve: {expr}",
        "assistant": f"####{val}"
    }

# -------------------------------
# TWO-OPERATION SAMPLE (2 steps)
# -------------------------------
def gen_two_ops():
    # numbers
    A = random_number()
    B = random_number()
    C = random_number()

    # operations
    op1 = random.choice(ops)
    op2 = random.choice(ops)

    # randomly choose pattern 1 or 2
    pattern = random.choice([1, 2])

    # ------------------------------------------------------------
    # PATTERN 1: ((A op1 B) op2 C)
    # ------------------------------------------------------------
    if pattern == 1:
        expr = f"( ( {A} {op1} {B} ) {op2} {C} )"

        # Step 1
        step1_expr = f"{A} {op1} {B}"
        step1_val = safe_eval(step1_expr)
        if step1_val is None:
            return gen_two_ops()

        # Step 2
        step2_expr = f"{step1_val} {op2} {C}"
        step2_val = safe_eval(step2_expr)
        if step2_val is None:
            return gen_two_ops()

        return {
            "user": f"Solve: {expr}",
            "assistant":
                f"{step1_expr} = {step1_val}\n"
                f"{step2_expr} = {step2_val}\n\n"
                f"####{step2_val}"
        }

    # ------------------------------------------------------------
    # PATTERN 2: (A op1 (B op2 C))
    # ------------------------------------------------------------
    else:
        expr = f"( {A} {op1} ( {B} {op2} {C} ) )"

        # Step 1
        step1_expr = f"{B} {op2} {C}"
        step1_val = safe_eval(step1_expr)
        if step1_val is None:
            return gen_two_ops()

        # Step 2
        step2_expr = f"{A} {op1} {step1_val}"
        step2_val = safe_eval(step2_expr)
        if step2_val is None:
            return gen_two_ops()

        return {
            "user": f"Solve: {expr}",
            "assistant":
                f"{step1_expr} = {step1_val}\n"
                f"{step2_expr} = {step2_val}\n\n"
                f"####{step2_val}"
        }

# -------------------------------
# WRITE DATASET
# -------------------------------
with open(OUTPUT_FILE, "w") as f:
    # 10,000 one-operation samples
    for _ in range(ONE_OP_SAMPLES):
        f.write(json.dumps(gen_one_op()) + "\n")

    # 5,000 two-operation samples with steps
    # for _ in range(TWO_OP_SAMPLES):
    #     f.write(json.dumps(gen_two_ops()) + "\n")

print("Dataset saved to:", OUTPUT_FILE)