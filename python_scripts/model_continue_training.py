"""
Vision-Language Model (VLM) Fine-tuning Script
------------------------------------------------
This script fine-tunes the SmolVLM-256M-Instruct model on the PlotQA dataset 
for chart interpretation and question answering.
------------------------------------------------
"""

# ============================================================================ #
# 1. ENVIRONMENT SETUP
# ============================================================================ #
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    Idefics3ForConditionalGeneration,
    AutoProcessor,
)
from dataclasses import dataclass
from typing import Any, Dict, List
from trl import SFTTrainer, SFTConfig
from collections import Counter
import numpy as np
import wandb
from tqdm import tqdm
import gc
import time

# ============================================================================ #
# 2. DEVICE SETUP
# ============================================================================ #
device = torch.cuda.current_device()

print("=" * 60)
print("CUDA DEVICE INFORMATION")
print("=" * 60)
print(f"CUDA Available : {torch.cuda.is_available()}")
print(f"Current Device : {torch.cuda.current_device()}")
print(f"Device Name    : {torch.cuda.get_device_name(torch.cuda.current_device())}")
print("=" * 60)

# ============================================================================ #
# 3. HUGGING FACE AUTHENTICATION
# ============================================================================ #
print("\n[1/9] Logging into Hugging Face...")
login("SECRET_HF_TOKEN")  # Replace with your actual token or use environment variable
print("✓ Successfully logged into Hugging Face")

# ============================================================================ #
# 4. DATASET LOADING AND CONFIGURATION
# ============================================================================ #
print("\n[2/9] Loading dataset...")
dataset_id = "Abd223653/Plot-QA-V1"

# Load datasets in streaming mode for memory efficiency
train_dataset = load_dataset(dataset_id, split="train", streaming=True)

# Define dataset subset boundaries
TRAIN_START, TRAIN_END = 23400, 38400

print(f"Training samples : {TRAIN_START} to {TRAIN_END}")

# Create subsets
train_set = train_dataset.skip(TRAIN_START).take(TRAIN_END)

# Shuffle training data for better generalization
print("Shuffling training data...")
train_set = train_set.shuffle(seed=42, buffer_size=10000)
print("✓ Dataset loaded and configured")

# ============================================================================ #
# 5. PROCESSOR INITIALIZATION
# ============================================================================ #
print("\n[3/9] Loading processor...")
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
print("✓ Processor loaded successfully")

# ============================================================================ #
# 6. SYSTEM PROMPT
# ============================================================================ #
system_message = """
You are a Vision-Language Model specialized in interpreting chart and plot images.
Analyze the chart carefully and answer the given question concisely (usually a single word, number, or short phrase).
Use both visual information (values, colors, labels) and simple reasoning (e.g., finding averages, differences, trends).
Do not rely on any external or prior knowledge — all answers must come from interpreting the chart itself.
"""

# ============================================================================ #
# 7. DATA FORMATTING FUNCTION
# ============================================================================ #
def format_data_train(sample):
    """
    Converts a dataset sample into a chat-style conversation for the model.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": f"Question Template: {sample['template']}\n"
                            f"Plot Type: {sample['type']}\n"
                            f"Question: {sample['question_string']}",
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False)

    return {"text": text, "images": [sample["image"]]}

# ============================================================================ #
# 8. DATASET FORMATTING
# ============================================================================ #
print("\n[4/9] Formatting datasets...")
print("Formatting training set...")
formatted_train_set = train_set.map(format_data_train, remove_columns=train_set.features)

# Manually free up memory
gc.collect()
torch.cuda.empty_cache()

# ============================================================================ #
# 9. CUSTOM DATA COLLATOR
# ============================================================================ #
@dataclass
class VLMCollator:
    """
    Collates text-image pairs into model inputs with labels.
    Handles padding, tokenization, and label masking.
    """
    processor: Any

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [ex["text"] for ex in examples]
        images = [ex["images"] for ex in examples]

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        labels = batch["input_ids"].clone()

        # Mask all tokens before assistant response (only learn assistant output)
        for i in range(len(labels)):
            response_start = (batch["input_ids"][i] == 198).nonzero(as_tuple=True)[0]
            if len(response_start) > 0:
                labels[i, : response_start[-2]] = -100  # Mask prompt tokens

        batch["labels"] = labels
        return batch

# ============================================================================ #
# 10. MODEL LOADING
# ============================================================================ #
print("\n[5/9] Loading base model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base model locally
model = Idefics3ForConditionalGeneration.from_pretrained(
    "/home/ie643_mindspring/models/smolvlm_256M",
    device_map={"": 0},
    dtype=torch.bfloat16,
    _attn_implementation="sdpa",
    local_files_only=True,
)
print("✓ Base model loaded")

# ============================================================================ #
# 11. LOAD PRETRAINED LORA ADAPTERS
# ============================================================================ #
print("\n[6/9] Loading LoRA adapters from checkpoint...")

adapter_path = "/home/ie643_mindspring/model_weights/training_8400to23400/checkpoint-938"
model.load_adapter(adapter_path, is_trainable=True)

# Verify trainable parameters
trainable_count = sum(p.requires_grad for _, p in model.named_parameters())
print(f"✓ LoRA adapters loaded and model moved to device")
print(f"Trainable parameter tensors: {trainable_count}")

gc.collect()
torch.cuda.empty_cache()

# ============================================================================ #
# 12. WEIGHTS & BIASES INITIALIZATION
# ============================================================================ #
print("\n[7/9] Initializing Weights & Biases...")
wandb.init(
    project="IE643_SmolVLM_Finetuning",
    name="smolvlm-256M-train-run-23400to38400"
)
print("✓ W&B initialized")

# ============================================================================ #
# 13. TRAINING CONFIGURATION
# ============================================================================ #
print("\n[8/9] Setting up training configuration...")

training_args = SFTConfig(
    output_dir="/home/ie643_mindspring/model_weights/training_23400to38400",
    max_steps=938,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=1.92e-7,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=75,
    save_total_limit=1,
    optim="adamw_torch",
    bf16=True,
    push_to_hub=False,
    max_grad_norm=1.0,
    report_to="wandb",
    max_length=1024,
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
    do_train=True
)
print("✓ Training configuration complete")

# ============================================================================ #
# 14. TRAINER INITIALIZATION
# ============================================================================ #
print("\n[9/9] Initializing trainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_train_set,
    processing_class=processor,
    data_collator=VLMCollator(processor=processor),
)

print("✓ Trainer initialized successfully")

# ============================================================================ #
# 15. TRAINING EXECUTION
# ============================================================================ #
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"Total steps           : {training_args.max_steps}")
print(f"Per-device batch size : {training_args.per_device_train_batch_size}")
print(f"Grad accumulation     : {training_args.gradient_accumulation_steps}")
print(f"Effective batch size  : {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print("=" * 60 + "\n")

trainer.train()

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
print(f"Model checkpoints saved to: {training_args.output_dir}")
print("=" * 60)
