import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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


def calculate_score_numerical_2(y_true, y_pred, eps=1e-7):
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
    "C:\\College Study\\Semester 5\\IE643\\Project\\Code\\results\\csv\\model_validation_streamed_training_stage2_image_text_part3.csv"
)

# Categorize samples into three types
df_yes_no = df_eval[df_eval["type"] == "yes_no"].reset_index(drop=True)
df_yes_no_filtered = df_yes_no[df_yes_no["model_answer"].str.contains("####", na=False)]
df_yes_no_filtered["true_answer"] = df_yes_no_filtered["true_answer"].apply(lambda x: x.strip().lower())
df_yes_no_filtered["model_answer"] = df_yes_no_filtered["model_answer"].apply(
    lambda x: x.split("####")[1].strip().lower() if "####" in x else x.strip().lower()
)

df_numeric = df_eval[df_eval["type"] == "numeric"].reset_index(drop=True)
df_numeric_filtered = df_numeric[df_numeric["model_answer"].str.contains("####", na=False)]
df_numeric_filtered["model_answer"] = df_numeric_filtered["model_answer"].apply(
    lambda x: x.split("####")[1].strip().lower()
)

df_textual = df_eval[df_eval["type"] == "textual"].reset_index(drop=True)
df_textual_filtered = df_textual[df_textual["model_answer"].str.contains("####", na=False)]
df_textual_filtered["model_answer"] = df_textual_filtered["model_answer"].apply(
    lambda x: x.split("####")[1].strip()
)

df_word_problem = df_eval[df_eval["type"] == "word_problem"].reset_index(drop=True)
df_word_problem_filtered = df_word_problem[df_word_problem["model_answer"].str.contains("####", na=False)]
df_word_problem_filtered["model_answer"] = df_word_problem_filtered["model_answer"].apply(
    lambda x: x.split("####")[1].strip()
)

df_arithmetic = df_eval[df_eval["type"] == "arithmetic"].reset_index(drop=True)
df_arithmetic_filtered = df_arithmetic[df_arithmetic["model_answer"].str.contains("####", na=False)]
df_arithmetic_filtered["model_answer"] = df_arithmetic_filtered["model_answer"].apply(
    lambda x: x.split("####")[1].strip()
)

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
logger.info(
    f"  - Word Problem Questions: {len(df_word_problem):,} samples ({100*len(df_word_problem)/len(df_eval):.1f}%)"
)
logger.info(
    f"  - Arithmetic Questions: {len(df_arithmetic):,} samples ({100*len(df_arithmetic)/len(df_eval):.1f}%)"
)

# ============================================================================
# STAGE 9: CALCULATE CATEGORY-WISE SCORES
# ============================================================================
logger.info("\n[3/4] Computing scores by question category...")

# Calculate Yes/No accuracy using binary matching
score_yes_no = (
    calculate_score_yes_no(df_yes_no_filtered["true_answer"], df_yes_no_filtered["model_answer"])
    if len(df_yes_no_filtered) > 0
    else 0.0
)
logger.info(f"✓ Yes/No score calculated: {score_yes_no:.4f}")

# Calculate Numerical accuracy using relative error (within 5% tolerance)
score_numerical = (
    calculate_score_numerical_2(
        pd.to_numeric(df_numeric_filtered["true_answer"], errors="coerce"),
        pd.to_numeric(df_numeric_filtered["model_answer"], errors="coerce"),
    )
    if len(df_numeric_filtered) > 0
    else 0.0
)
logger.info(f"✓ Numerical score calculated: {score_numerical:.4f}")

# Calculate Textual accuracy using case-insensitive substring matching
score_textual = (
    calculate_score_textual(df_textual_filtered["true_answer"], df_textual_filtered["model_answer"])
    if len(df_textual_filtered) > 0
    else 0.0
)
logger.info(f"✓ Textual score calculated: {score_textual:.4f}")

score_word_problem = (
    calculate_score_numerical_2(
        pd.to_numeric(df_word_problem_filtered["true_answer"], errors="coerce"),
        pd.to_numeric(df_word_problem_filtered["model_answer"], errors="coerce"),
    )
    if len(df_word_problem_filtered) > 0
    else 0.0
)

logger.info(f"✓ Word Problem score calculated: {score_word_problem:.4f}")

score_arithmetic = (
    calculate_score_numerical_2(
        pd.to_numeric(df_arithmetic_filtered["true_answer"], errors="coerce"),
        pd.to_numeric(df_arithmetic_filtered["model_answer"], errors="coerce"),
    )
    if len(df_arithmetic_filtered) > 0
    else 0.0
)

logger.info(f"✓ Arithmetic score calculated: {score_arithmetic:.4f}")

# Calculate weighted final score: weighted average across all categories
total_samples = len(df_yes_no_filtered) + len(df_numeric_filtered) + len(df_textual_filtered) + len(df_word_problem_filtered) + len(df_arithmetic_filtered)
final_score = (
    (
        (
            score_yes_no * len(df_yes_no_filtered)
            + score_numerical * len(df_numeric_filtered)
            + score_textual * len(df_textual_filtered)
            + score_word_problem * len(df_word_problem_filtered)
            + score_arithmetic * len(df_arithmetic_filtered)
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
logger.info(f"✓ Yes/No Score        : {score_yes_no:.4f} ({len(df_yes_no_filtered):,} samples)")
logger.info(f"✓ Numerical Score     : {score_numerical:.4f} ({len(df_numeric_filtered):,} samples)")
logger.info(f"✓ Textual Score       : {score_textual:.4f} ({len(df_textual_filtered):,} samples)")
logger.info(f"✓ Word Problem Score  : {score_word_problem:.4f} ({len(df_word_problem_filtered):,} samples)")
logger.info(f"✓ Arithmetic Score    : {score_arithmetic:.4f} ({len(df_arithmetic_filtered):,} samples)")
logger.info("=" * 60)
logger.info(f"✓ Final Validation Score : {final_score:.4f} ({total_samples:,} samples)")
logger.info("=" * 60)

# ============================================================================
# STAGE 10: VISUALIZATION - CONFUSION MATRIX
# ============================================================================
logger.info("\n[4/4] Generating visualizations...")

# Generate confusion matrix for Yes/No questions (categorical classification performance)
if len(df_yes_no_filtered) > 0:
    logger.info("\nGenerating confusion matrix for Yes/No questions...")
    cf_matrix = confusion_matrix(df_yes_no_filtered["true_answer"], df_yes_no_filtered["model_answer"])

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
    output_viz_path = "conf_matrix_validation_streamed_training_stage2_image_text.png"
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