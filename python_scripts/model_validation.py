# ===============================================
# 🧠 VALIDATION SCRIPT FOR SmolVLM-256M MODEL
# ===============================================

import os
import gc
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Idefics3ForConditionalGeneration,
)

# -----------------------------------------------
# ⚙️ 1. Environment and Device Setup
# -----------------------------------------------
print("\n" + "=" * 60)
print("STAGE 1: ENVIRONMENT & DEVICE SETUP")
print("=" * 60)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✓ CUDA Available        : {torch.cuda.is_available()}")
print(f"✓ Device               : {device}")
if torch.cuda.is_available():
    print(f"✓ GPU Name             : {torch.cuda.get_device_name(device)}")
    print(f"✓ GPU Memory           : {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

# Login to Hugging Face Hub (ensure token is valid)
print("\nLogging into Hugging Face Hub...")
login("SECRET_HF_TOKEN")  # Replace with your actual token or use environment variable
print("✓ Successfully logged into Hugging Face")
print("=" * 60)

# -----------------------------------------------
# 📂 2. Load Dataset (Streaming for Efficiency)
# -----------------------------------------------
print("\n" + "=" * 60)
print("STAGE 2: DATASET LOADING")
print("=" * 60)

dataset_id = "Abd223653/Plot-QA-V1"
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

print(f"\nDataset ID             : {dataset_id}")
print(f"Model ID               : {model_id}")

# Load processor (handles both text & image inputs)
print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(model_id)

# Fix padding for decoder-only architecture
# Set padding_side to 'left' for correct generation with batched inputs
if hasattr(processor, 'tokenizer'):
    processor.tokenizer.padding_side = 'left'
    print(f"✓ Tokenizer padding_side set to: {processor.tokenizer.padding_side}")
else:
    print("⚠️  Warning: Could not set tokenizer padding_side")

print("✓ Processor loaded successfully")

# Load validation split in streaming mode
print("\nLoading validation dataset in streaming mode...")
val_dataset = load_dataset(dataset_id, split="test", streaming=True)
print("✓ Validation dataset loaded (streaming)")

# Define validation subset range
VAL_START = 0
VAL_END = 500

# Slice subset for validation
print(f"\nSlicing dataset from {VAL_START} to {VAL_END}...")
val_set = val_dataset.skip(VAL_START).take(VAL_END - VAL_START)
print(f"✓ Dataset sliced: {VAL_END - VAL_START} samples selected")
print("=" * 60)

# -----------------------------------------------
# 🧾 3. Define System Message and Data Formatting
# -----------------------------------------------
system_message = """
You are a Vision-Language Model specialized in interpreting chart and plot images.
Analyze the chart carefully and answer the given question concisely (usually a single word, number, or short phrase).
Use both visual information (values, colors, labels) and simple reasoning (e.g., finding averages, differences, trends) based on the chart data.
Do not rely on any external or prior knowledge — all answers must come from interpreting the chart itself.
"""

def format_data_val(sample):
    """
    Formats raw dataset samples into model-compatible input format.

    Args:
        sample (dict): A single dataset sample with keys like 'image', 'question_string', and 'answer'.
    
    Returns:
        dict: Contains formatted 'text', 'images', and metadata (question, answer, image_index).
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {sample['question_string']}"},
            ],
        },
    ]

    # Convert structured messages into model-readable chat format
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return {
        "text": text,
        "images": [sample["image"]],
        "question": sample.get("question_string", ""),
        "answer": sample.get("answer", ""),
        "image_index": sample.get("image_index", -1),
    }

# Apply formatting function to validation data
formatted_val_set = val_set.map(format_data_val, remove_columns=val_set.features)

print("\n" + "=" * 60)
print("STAGE 3: DATA FORMATTING")
print("=" * 60)
print("✓ Data formatting function applied")
print(f"✓ Formatted dataset ready for collation")
print("=" * 60)

# -----------------------------------------------
# 🧩 4. Define Custom Data Collator
# -----------------------------------------------
@dataclass
class VLMCollator:
    """
    Custom collator for batching vision-language inputs during inference.
    Optimized to skip unnecessary label creation since labels are only needed for training.
    """
    processor: Any

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [ex["text"] for ex in examples]
        images = [ex["images"] for ex in examples]

        # Tokenize & preprocess with left-padding for decoder-only models
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side='left',  # Required for decoder-only architecture
        )

        # Preserve metadata (question, answer, image_index)
        batch["questions"] = [ex["question"] for ex in examples]
        batch["answers"] = [ex["answer"] for ex in examples]
        batch["image_indexes"] = [ex["image_index"] for ex in examples]
        
        return batch

print("\n" + "=" * 60)
print("STAGE 4: DATA COLLATOR INITIALIZATION")
print("=" * 60)
print("✓ VLMCollator class defined and ready")
print("=" * 60)

# -----------------------------------------------
# 🧠 5. Load Model and Adapter
# -----------------------------------------------
# Enable 4-bit quantization for reduced VRAM usage (inference-friendly)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("\n" + "=" * 60)
print("STAGE 5: MODEL LOADING & QUANTIZATION")
print("=" * 60)
print("\nQuantization Config:")
print("✓ 4-bit quantization enabled")
print("✓ Double quantization enabled")
print("✓ Quantization type: nf4")
print("✓ Compute dtype: bfloat16")

# Load pretrained base model with quantization
print("\nLoading pretrained base model...")
model = Idefics3ForConditionalGeneration.from_pretrained(
    "/home/ie643_mindspring/models/smolvlm_256M",
    device_map={"": 0},
    dtype=torch.bfloat16,
    # quantization_config=bnb_config,  # Enable 4-bit quantization
    _attn_implementation="sdpa",
    local_files_only=True,
)
print("✓ Base model loaded successfully")

# Print model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Architecture:")
print(f"✓ Total Parameters: {total_params:,}")
print(f"✓ Trainable Parameters: {trainable_params:,}")

# Load LoRA adapter checkpoint
print("\nLoading LoRA adapter checkpoint...")
adapter_path = "/home/ie643_mindspring/model_weights/training_23400to38400/checkpoint-938"
model.load_adapter(adapter_path)
print(f"✓ LoRA adapter loaded from: {adapter_path}")

model.eval()  # Ensure evaluation mode
print("✓ Model set to evaluation mode")
print("=" * 60)

# -----------------------------------------------
# 🔍 6. Run Inference on Validation Set (Streaming CSV Save)
# -----------------------------------------------
import csv

preds = []
questions = []
answers = []
image_indexes = []

print("\n" + "=" * 60)
print("STAGE 6: INFERENCE ON VALIDATION SET (STREAMING OUTPUT)")
print("=" * 60)

batch_size = 16
batch_items = []
collator = VLMCollator(processor=processor)

output_path = "/home/ie643_mindspring/results/csv/model_validation_streamed_test_set-training-23400to38400-1to500.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Prepare CSV and write header once
if not os.path.exists(output_path):
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index", "question", "true_answer", "model_answer"])

print(f"\nInference Configuration:")
print(f"✓ Batch size: {batch_size}")
print(f"✓ Output file: {output_path}")
print(f"✓ Writing results incrementally (stream mode)\n")

start_time = time.time()
batch_count = 0
processed_samples = 0

for item in tqdm(formatted_val_set, desc="Generating Responses"):
    batch_items.append(item)
    
    if len(batch_items) == batch_size:
        batch = collator(batch_items)
        batch_questions = batch.pop("questions")
        batch_answers = batch.pop("answers")
        batch_image_indexes = batch.pop("image_indexes")

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **batch,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=True,
            )

        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Write results of this batch to CSV
        with open(output_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for i, full_text in enumerate(generated_texts):
                assistant_idx = full_text.rfind("Assistant: ")
                prediction = full_text[assistant_idx + len("Assistant: "):].strip() if assistant_idx != -1 else ""
                writer.writerow([
                    batch_image_indexes[i],
                    batch_questions[i],
                    batch_answers[i],
                    prediction
                ])

        processed_samples += len(batch_items)
        batch_items = []
        batch_count += 1
        
        # Cleanup
        del batch, generated_ids, generated_texts
        gc.collect()
        torch.cuda.empty_cache()

        if batch_count % 10 == 0:
            print(f"✓ Processed {processed_samples:,} samples so far...")

# Handle remaining items
if batch_items:
    batch = collator(batch_items)
    batch_questions = batch.pop("questions")
    batch_answers = batch.pop("answers")
    batch_image_indexes = batch.pop("image_indexes")

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **batch,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True,
        )

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    with open(output_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i, full_text in enumerate(generated_texts):
            assistant_idx = full_text.rfind("Assistant: ")
            prediction = full_text[assistant_idx + len("Assistant: "):].strip() if assistant_idx != -1 else ""
            writer.writerow([
                batch_image_indexes[i],
                batch_questions[i],
                batch_answers[i],
                prediction
            ])

    processed_samples += len(batch_items)
    batch_count += 1

elapsed_time = time.time() - start_time
print("-" * 60)
print(f"✓ Inference completed successfully!")
print(f"✓ Total processed samples: {processed_samples:,}")
print(f"✓ Total batches: {batch_count}")
print(f"✓ Total time: {elapsed_time:.2f}s ({elapsed_time / processed_samples:.2f}s/sample)")
print(f"✓ Results saved incrementally at: {output_path}")
print("=" * 60)

# -----------------------------------------------
# 9. EVALUATION METRICS & SCORING FUNCTIONS
# -----------------------------------------------
print("\n" + "=" * 60)
print("STAGE 9: EVALUATION METRICS & SCORING")
print("=" * 60)

print("\n[1/4] Defining scoring functions...")

def calculate_score_yes_no(y_true, y_pred):
    """
    Calculate accuracy score for Yes/No questions.
    
    Args:
        y_true (pd.Series): True labels
        y_pred (pd.Series): Predicted labels
    
    Returns:
        float: Accuracy score (0-1)
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

print("✓ Scoring functions defined successfully")

# -----------------------------------------------
# 10. DATA CATEGORIZATION & EVALUATION
# -----------------------------------------------
print("\n[2/4] Categorizing predictions by question type...")

# Create evaluation dataframe from predictions
df_eval = pd.read_csv('/home/ie643_mindspring/results/csv/model_validation_streamed_test_set_training_23400to38400.csv')

# Categorize samples into three types
mask_yes_no = df_eval['true_answer'].isin(['Yes', 'No'])
mask_numeric = pd.to_numeric(df_eval['true_answer'], errors='coerce').notna()
mask_textual = ~(mask_yes_no | mask_numeric)

# Split data by question type
df_yes_no = df_eval[mask_yes_no].copy()
df_numeric = df_eval[mask_numeric].copy()
df_textual = df_eval[mask_textual].copy()

print(f"✓ Dataset categorized:")
print(f"  - Yes/No Questions: {len(df_yes_no):,} samples ({100*len(df_yes_no)/len(df_eval):.1f}%)")
print(f"  - Numerical Questions: {len(df_numeric):,} samples ({100*len(df_numeric)/len(df_eval):.1f}%)")
print(f"  - Textual Questions: {len(df_textual):,} samples ({100*len(df_textual)/len(df_eval):.1f}%)")

# -----------------------------------------------
# 11. CALCULATE CATEGORY-WISE SCORES
# -----------------------------------------------
print("\n[3/4] Computing scores by question category...")

# Calculate Yes/No accuracy
score_yes_no = calculate_score_yes_no(
    df_yes_no['true_answer'], 
    df_yes_no['model_answer']
) if len(df_yes_no) > 0 else 0.0
print(f"✓ Yes/No score calculated: {score_yes_no:.4f}")

# Calculate Numerical relative error
score_numerical = calculate_score_numerical(
    pd.to_numeric(df_numeric['true_answer'], errors='coerce'),
    pd.to_numeric(df_numeric['model_answer'], errors='coerce')
) if len(df_numeric) > 0 else 0.0
print(f"✓ Numerical score calculated: {score_numerical:.4f}")

# Calculate Textual accuracy
score_textual = calculate_score_textual(
    df_textual['true_answer'], 
    df_textual['model_answer']
) if len(df_textual) > 0 else 0.0
print(f"✓ Textual score calculated: {score_textual:.4f}")

# Calculate weighted final score
total_samples = len(df_eval)
final_score = (
    (score_yes_no * len(df_yes_no) + 
     score_numerical * len(df_numeric) + 
     score_textual * len(df_textual)) / total_samples
) if total_samples > 0 else 0.0

print(f"\n✓ Final weighted score calculated: {final_score:.4f}")

print("\n" + "-" * 60)
print("VALIDATION SCORES BY CATEGORY")
print("-" * 60)
print(f"✓ Yes/No Score        : {score_yes_no:.4f} ({len(df_yes_no):,} samples)")
print(f"✓ Numerical Score     : {score_numerical:.4f} ({len(df_numeric):,} samples)")
print(f"✓ Textual Score       : {score_textual:.4f} ({len(df_textual):,} samples)")
print("-" * 60)
print(f"✓ Final Validation Score : {final_score:.4f} ({total_samples:,} samples)")
print("-" * 60)

# -----------------------------------------------
# 12. VISUALIZATION - CONFUSION MATRIX
# -----------------------------------------------
print("\n[4/4] Generating visualizations...")

# Generate confusion matrix for Yes/No questions
if len(df_yes_no) > 0:
    print("\nGenerating confusion matrix for Yes/No questions...")
    cf_matrix = confusion_matrix(df_yes_no['true_answer'], df_yes_no['model_answer'])
    
    # Create heatmap visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cf_matrix / np.sum(cf_matrix),
        annot=True,
        fmt='.2%',
        cmap='Blues',
        cbar_kws={'label': 'Proportion'},
        square=True
    )
    
    # Add axis labels and title
    plt.xlabel("Predicted Labels", fontsize=12, fontweight='bold')
    plt.ylabel("True Labels", fontsize=12, fontweight='bold')
    plt.title("Confusion Matrix - Yes/No Questions", fontsize=14, fontweight='bold')
    
    # Save visualization
    output_viz_path = "/home/ie643_mindspring/results/plots/conf_matrix_validation_streamed_training-with-masking-with-connector-link-lora8-1to1000.png"
    plt.savefig(output_viz_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_viz_path}")
    
    # Verify file was saved
    if os.path.exists(output_viz_path):
        viz_size_kb = os.path.getsize(output_viz_path) / 1024
        print(f"✓ Visualization file size: {viz_size_kb:.2f} KB")
    
    plt.close()
else:
    print("⚠️  No Yes/No questions found for confusion matrix")

print("\n✨ Evaluation completed successfully!")
print("=" * 60 + "\n")
