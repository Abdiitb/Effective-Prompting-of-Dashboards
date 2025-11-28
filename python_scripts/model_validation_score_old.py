import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# STAGE 7: EVALUATION METRICS & SCORING FUNCTIONS
# ============================================================================
logger.info("\n" + "=" * 60)
logger.info("STAGE 7: EVALUATION METRICS & SCORING")
logger.info("=" * 60)

logger.info("\n[1/4] Defining scoring functions...")

def is_float_string(value):
    """
    Check if a value (as string) contains a float number.
    
    Handles various float formats:
    - Standard: 1.5, -2.3, 0.001
    - Scientific: 1e-5, 2.5e3
    - With spaces: " 1.5 " (after stripping)
    
    Args:
        value: Any value to check
    
    Returns:
        bool: True if value is or contains a float number, False otherwise
    """
    if not isinstance(value, str):
        return False
    
    value = value.strip()
    
    # Pattern to match float numbers (including scientific notation)
    # Matches: 1.5, -1.5, 1e-5, -1.5e3, .5, -.5, etc.
    float_pattern = r'^-?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$'
    
    return bool(re.match(float_pattern, value))

def calculate_score_yes_no(y_true, y_pred):
    """
    Calculate exact match accuracy for Yes/No questions.
    
    Compares predicted labels with ground truth labels for simple binary classification.

    Args:
        y_true (pd.Series): True labels (Yes/No)
        y_pred (pd.Series): Predicted labels (Yes/No)

    Returns:
        float: Accuracy score (0-1, higher is better)
    """
    return (y_true == y_pred).sum() / len(y_true)


def calculate_score_numerical(y_true, y_pred, eps=1e-7):
    """
    Calculate relative error score for numerical questions.

    Args:
        y_true (pd.Series): True numerical values
        y_pred (pd.Series): Predicted numerical values
        eps (float): Small value to avoid division by zero

    Returns:
        float: Mean relative error score (0-1)
    """
    rel_err = 1 - abs((y_true - y_pred) / (y_true + eps))
    rel_err = rel_err[rel_err >= 0]
    score = rel_err.sum() / len(rel_err) if len(rel_err) > 0 else 0
    return score


def calculate_score_numerical_2(y_true, y_pred):
    """
    Calculate relative error score for numerical questions.

    Args:
        y_true (pd.Series): True numerical values
        y_pred (pd.Series): Predicted numerical values
        eps (float): Small value to avoid division by zero

    Returns:
        float: Mean relative error score (0-1)
    """
    correct_indexes = (y_pred <= y_true * 1.05) & (y_pred >= y_true * 0.95)
    score = correct_indexes.sum() / len(y_true) if len(y_true) > 0 else 0
    return score


def calculate_score_textual(y_true, y_pred):
    """
    Calculate accuracy score for textual questions (case-insensitive).

    Args:
        y_true (pd.Series): True text labels
        y_pred (pd.Series): Predicted text labels

    Returns:
        float: Accuracy score (0-1)
    """
    y_true_ = y_true.apply(lambda x: str(x).lower().strip())
    y_pred_ = y_pred.apply(lambda x: str(x).lower().strip())
    return (y_true_ == y_pred_).sum() / len(y_true_)

logger.info("✓ Scoring functions defined successfully")

# ============================================================================
# STAGE 8: DATA CATEGORIZATION & EVALUATION
# ============================================================================
logger.info("\n[2/4] Categorizing predictions by question type...")

# Load evaluation results from CSV file
df_eval = pd.read_csv(
    "C:\\College Study\\Semester 5\\IE643\\Project\\Code\\results\\csv\\model_validation_streamed_23400to38400.csv"
)

# Categorize samples into three types
df_yes_no = df_eval[(df_eval["true_answer"].str.lower() == "yes") | (df_eval["true_answer"].str.lower() == "no")].reset_index(drop=True)
df_yes_no["model_answer"] = df_yes_no["model_answer"].str.lower().str.strip().str.replace(".", "", regex=False).str.replace("no\nanswer: ", "", regex=False)
df_yes_no["true_answer"] = df_yes_no["true_answer"].str.lower().str.strip().str.replace(".", "", regex=False)

df_numeric = df_eval.filter(
    axis=0,
    items=[
        idx
        for idx, ans in df_eval["true_answer"].items()
        if is_float_string(str(ans).strip())
    ],
).reset_index(drop=True)

df_textual = df_eval.filter(
    axis=0,
    items=[
        idx
        for idx, ans in df_eval["true_answer"].items()
        if not is_float_string(str(ans).strip())
        and str(ans).strip().lower() not in ["yes", "no"]
    ],
).reset_index(drop=True)
df_textual["model_answer"] = df_textual["model_answer"].str.lower().str.strip().str.replace(".", "", regex=False)
df_textual["true_answer"] = df_textual["true_answer"].str.lower().str.strip().str.replace(".", "", regex=False)

# Log categorized dataset composition by question type
logger.info(f"✓ Dataset categorized:")
logger.info(
    f"  - Yes/No Questions: {len(df_yes_no):,} samples ({100*len(df_yes_no)/len(df_eval):.1f}%)"
)
logger.info(
    f"  - Numerical Questions: {len(df_numeric):,} samples ({100*len(df_numeric)/len(df_eval):.1f}%)"
)
logger.info(
    f"  - Textual Questions: {len(df_textual):,} samples ({100*len(df_textual)/len(df_eval):.1f}%)"
)

# ============================================================================
# STAGE 9: CALCULATE CATEGORY-WISE SCORES
# ============================================================================
logger.info("\n[3/4] Computing scores by question category...")

# Calculate Yes/No accuracy using binary matching
score_yes_no = (
    calculate_score_yes_no(df_yes_no["true_answer"], df_yes_no["model_answer"])
    if len(df_yes_no) > 0
    else 0.0
)
logger.info(f"✓ Yes/No score calculated: {score_yes_no:.4f}")

# Calculate Numerical accuracy using relative error (within 5% tolerance)
score_numerical = (
    calculate_score_numerical_2(
        pd.to_numeric(df_numeric["true_answer"], errors="coerce"),
        pd.to_numeric(df_numeric["model_answer"], errors="coerce"),
    )
    if len(df_numeric) > 0
    else 0.0
)
logger.info(f"✓ Numerical score calculated: {score_numerical:.4f}")

# Calculate Textual accuracy using case-insensitive substring matching
score_textual = (
    calculate_score_textual(df_textual["true_answer"], df_textual["model_answer"])
    if len(df_textual) > 0
    else 0.0
)
logger.info(f"✓ Textual score calculated: {score_textual:.4f}")

# Calculate weighted final score: weighted average across all categories
total_samples = len(df_eval)
final_score = (
    (
        (
            score_yes_no * len(df_yes_no)
            + score_numerical * len(df_numeric)
            + score_textual * len(df_textual)
        )
        / total_samples
    )
    if total_samples > 0
    else 0.0
)


# Log final scores with category breakdown
logger.info(f"\n✓ Final weighted score calculated: {final_score:.4f}")
logger.info("\n" + "=" * 60)
logger.info("VALIDATION SCORES BY CATEGORY")
logger.info("=" * 60)
logger.info(f"✓ Yes/No Score        : {score_yes_no:.4f} ({len(df_yes_no):,} samples)")
logger.info(f"✓ Numerical Score     : {score_numerical:.4f} ({len(df_numeric):,} samples)")
logger.info(f"✓ Textual Score       : {score_textual:.4f} ({len(df_textual):,} samples)")
logger.info("=" * 60)
logger.info(f"✓ Final Validation Score : {final_score:.4f} ({total_samples:,} samples)")
logger.info("=" * 60)

# ============================================================================
# STAGE 10: VISUALIZATION - CONFUSION MATRIX
# ============================================================================
logger.info("\n[4/4] Generating visualizations...")

# Generate confusion matrix for Yes/No questions (categorical classification performance)
if len(df_yes_no) > 0:
    logger.info("\nGenerating confusion matrix for Yes/No questions...")
    print(df_yes_no["model_answer"].value_counts())
    cf_matrix = confusion_matrix(df_yes_no["true_answer"], df_yes_no["model_answer"])

    # Create heatmap visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cf_matrix / np.sum(cf_matrix),
        annot=True,
        fmt=".2%",
        cmap="Blues",
        cbar_kws={"label": "Proportion"},
        square=True,
    )

    # Add axis labels and title
    plt.xlabel("Predicted Labels", fontsize=12, fontweight="bold")
    plt.ylabel("True Labels", fontsize=12, fontweight="bold")
    plt.title("Confusion Matrix - Yes/No Questions", fontsize=14, fontweight="bold")

    # Save visualization to disk with high DPI for publication quality
    output_viz_path = "conf_matrix_validation_streamed_23400to38400.png"
    plt.savefig(output_viz_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Confusion matrix saved to: {output_viz_path}")

    # Log file statistics for verification
    if os.path.exists(output_viz_path):
        viz_size_kb = os.path.getsize(output_viz_path) / 1024
        logger.info(f"✓ Visualization file size: {viz_size_kb:.2f} KB")

    plt.close()
else:
    logger.warning("⚠️  No Yes/No questions found for confusion matrix")

# ============================================================================
# EVALUATION COMPLETE
# ============================================================================
logger.info("\n✨ Evaluation completed successfully!")
logger.info("=" * 60 + "\n")