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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✓ CUDA Available        : {torch.cuda.is_available()}")
print(f"✓ Device               : {device}")
if torch.cuda.is_available():
    print(f"✓ GPU Name             : {torch.cuda.get_device_name(device)}")
    print(f"✓ GPU Memory           : {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

# Login to Hugging Face Hub (ensure token is valid)
print("\nLogging into Hugging Face Hub...")
login("YOUR_HF_TOKEN")  # Replace with your actual token or use environment variable
print("✓ Successfully logged into Hugging Face")
print("=" * 60)

# -----------------------------------------------
# 📂 2. Load Dataset (Streaming for Efficiency)
# -----------------------------------------------
print("\n" + "=" * 60)
print("STAGE 2: DATASET LOADING")
print("=" * 60)

dataset_id = "openai/gsm8k"
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
val_set = load_dataset(dataset_id, "main", split="test", streaming=True)
print("✓ Validation dataset loaded (streaming)")

# Slice subset for validation
# print(f"\nSlicing dataset from {VAL_START} to {VAL_END}...")
# val_set = val_dataset.skip(VAL_START).take(VAL_END - VAL_START)
# print(f"✓ Dataset sliced: {VAL_END - VAL_START} samples selected")
# print("=" * 60)

# -----------------------------------------------
# 🧾 3. Define System Message and Data Formatting
# -----------------------------------------------
system_message = """
You are a reasoning assistant. Solve the problem step by step, showing all your thinking clearly. After completing the reasoning, provide the final answer following the token sequence "####". 
- Only include the final answer after "####".  
- The final answer can be numeric, text, or categorical depending on the problem.  
- Do not include any additional text after the answer.  
- Ensure all reasoning steps are explicit and easy to follow.
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
                {"type": "text", "text": f"Question: {sample['question']}"},
            ],
        },
    ]

    # Convert structured messages into model-readable chat format
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return {
        "text": text,
        "question": sample.get("question", ""),
        "answer": sample.get("answer", "")
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

        # Tokenize & preprocess with left-padding for decoder-only models
        batch = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            padding_side='left',  # Required for decoder-only architecture
        )

        # Preserve metadata (question, answer, image_index)
        batch["questions"] = [ex["question"] for ex in examples]
        batch["answers"] = [ex["answer"] for ex in examples]
        
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
adapter_path = "/home/ie643_mindspring/model_weights/training-language-head-gsm8k/checkpoint-468"
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

print("\n" + "=" * 60)
print("STAGE 6: INFERENCE ON VALIDATION SET (STREAMING OUTPUT)")
print("=" * 60)

batch_size = 32
batch_items = []
collator = VLMCollator(processor=processor)

output_path = "/home/ie643_mindspring/results/csv/model_validation_streamed_gsm8k_validation.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Prepare CSV and write header once
if not os.path.exists(output_path):
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "true_answer", "model_answer"])

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

        # Move batch tensors to the same device as the model
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **batch,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=True,
            )

        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Write results of this batch to CSV
        with open(output_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for i, full_text in enumerate(generated_texts):
                prediction = full_text.split("Assistant:")[-1]
                writer.writerow([
                    batch_questions[i],
                    batch_answers[i],
                    prediction
                ])

        processed_samples += len(batch_items)
        batch_items = []
        batch_count += 1

        if batch_count % 10 == 0:
            # Cleanup
            del batch, generated_ids, generated_texts
            gc.collect()
            torch.cuda.empty_cache()
            print(f"✓ Processed {processed_samples:,} samples so far...")

# Handle remaining items
if batch_items:
    batch = collator(batch_items)
    batch_questions = batch.pop("questions")
    batch_answers = batch.pop("answers")

    # Move batch tensors to the same device as the model
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
            prediction = full_text.split("Assistant:")[-1]
            writer.writerow([
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