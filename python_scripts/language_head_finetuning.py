"""
Vision-Language Model (VLM) Fine-tuning Script
Fine-tunes SmolVLM-256M-Instruct on PlotQA dataset for chart interpretation
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import torch
from huggingface_hub import login
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoProcessor
from dataclasses import dataclass
from typing import Any, Dict, List
from collections import Counter
from transformers import BitsAndBytesConfig, Idefics3ForConditionalGeneration, AutoProcessor, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import numpy as np
import gc
import time
import wandb
from tqdm import tqdm
from peft import PeftModel

# ============================================================================
# DEVICE SETUP AND VERIFICATION
# ============================================================================
device = torch.cuda.current_device()

print("=" * 60)
print("CUDA DEVICE INFORMATION")
print("=" * 60)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print("=" * 60)

# ============================================================================
# HUGGING FACE AUTHENTICATION
# ============================================================================
print("\n[1/9] Logging into Hugging Face...")
login("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")  # Replace with your actual token or use environment variable
print("✓ Successfully logged into Hugging Face")

# ============================================================================
# DATASET LOADING AND CONFIGURATION
# ============================================================================
print("\n[2/9] Loading dataset...")
dataset_id = "xuyige/ASDiv-Aug"

# Load datasets in streaming mode for memory efficiency
train_set_total = load_dataset(dataset_id, split="train")

train_set_non_plus = train_set_total.filter(lambda x: x['answer'].count('+') != 1)
train_set_plus = train_set_total.filter(lambda x: x['answer'].count('+') == 1).select(range(150))
random.seed(42)
indices = random.sample(range(len(train_set_plus)), 150)
train_set = concatenate_datasets([train_set_non_plus, train_set_plus.select(indices)])
train_set = train_set.shuffle(seed=42)

# ============================================================================
# PROCESSOR INITIALIZATION
# ============================================================================
print("\n[3/9] Loading processor...")
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
processor.eos_token = processor.tokenizer.eos_token
print("✓ Processor loaded successfully")

# ============================================================================
# SYSTEM PROMPT DEFINITION
# ============================================================================
system_message = """
You are an assistant that solves numerical and plot-based questions.
Perform calculations when needed, show clear step-by-step reasoning, and provide the final answer after a blank line in the format: ####<answer>.
For multi-step problems, show all steps.
For single-step problems, give the final answer with ####.
Respect parentheses and operation precedence.
"""

# ============================================================================
# DATA FORMATTING FUNCTION
# ============================================================================
def format_data_train(sample):
    """
    Formats raw dataset samples into chat template format.
    
    Args:
        sample: Dictionary containing 'image', 'template', 'type', 'question_string', 'answer'
    
    Returns:
        Dictionary with formatted 'text' and 'images' fields
    """
    # Construct multi-turn conversation with system, user, and assistant messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{sample['question']}"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample['answer'].replace('<<', '').replace('>>', '')}],
        },
    ]

    # Apply chat template to convert messages to model-specific format
    text = processor.apply_chat_template(messages, tokenize=False)

    return {
        "text": text,  # Formatted conversation text
    }

# ============================================================================
# DATASET FORMATTING
# ============================================================================
print("\n[4/9] Formatting datasets...")
print("Formatting training set...")
formatted_train_set = train_set.map(
    format_data_train, 
    remove_columns=train_set.features
)

print("✓ Datasets formatted successfully")

# ============================================================================
# CUSTOM DATA COLLATOR
# ============================================================================
@dataclass
class VLMCollator:
    """
    Custom collator for Vision-Language Model training.
    Handles batching of text and images, and creates labels for training.
    """
    processor: Any

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a batch of examples.
        
        Args:
            examples: List of dictionaries with 'text' and 'images' keys
        
        Returns:
            Batch dictionary with input_ids, attention_mask, pixel_values, and labels
        """
        # Extract texts and images from batch
        texts = [ex["text"] for ex in examples]

        # Process inputs with padding and convert to tensors
        batch = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True
        )

        # Create labels by copying input_ids (standard causal LM approach)
        labels = batch["input_ids"].clone()

        # Mask prompt tokens so model only learns to predict assistant responses
        for i in range(len(labels)):
            # Find newline tokens (42) to locate assistant response start
            response_start = (batch["input_ids"][i] == 42).nonzero(as_tuple=True)[0]
            if len(response_start) > 0:
                # Mask all tokens before the assistant's response with -100
                # -100 is ignored by PyTorch loss functions
                labels[i, : response_start[-1] + 2] = -100

        batch["labels"] = labels
        return batch

# ============================================================================
# MODEL LOADING WITH QUANTIZATION (OPTIONAL)
# ============================================================================
print("\n[5/9] Loading base model...")

# Quantization configuration (currently commented out)
# Enables 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Double quantization for better compression
    bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in bfloat16 for stability
)

# Load pre-trained VLM model from local directory
model = Idefics3ForConditionalGeneration.from_pretrained(
    "/home/ie643_mindspring/models/smolvlm_256M",
    device_map={"": 0},  # Load entire model on GPU 0
    dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    # quantization_config=bnb_config,  # Uncomment to enable quantization
    _attn_implementation="sdpa",  # Scaled Dot Product Attention (efficient)
    local_files_only=True  # Don't download, use local files only
)
print("✓ Base model loaded")

model = PeftModel.from_pretrained(model, "/home/ie643_mindspring/model_weights/training-language-head-finetuning-stage2-asdiv-re/checkpoint-262", is_trainable=True)

# ============================================================================
# LORA (PEFT) CONFIGURATION
# ============================================================================
print("\n[6/9] Configuring LoRA adapters...")


# # LoRA configuration for parameter-efficient fine-tuning
# peft_config = LoraConfig(
#     r=16,  # LoRA rank (higher = more parameters)
#     lora_alpha=32,  # LoRA scaling factor
#     lora_dropout=0.1,  # Dropout for regularization
#     target_modules=[  # Which modules to apply LoRA to
#         "down_proj", "o_proj", "k_proj", "q_proj", 
#         "gate_proj", "up_proj", "v_proj"
#     ],
#     use_dora=True,  # Use DoRA (Weight-Decomposed Low-Rank Adaptation)
#     init_lora_weights="gaussian",  # Gaussian initialization for LoRA weights
# )

# # Apply LoRA adapters to model
# model = get_peft_model(model, peft_config)

# # Display trainable parameters summary
# print("\n" + "=" * 60)
# model.print_trainable_parameters()
# print("=" * 60)

# # Count and display number of trainable parameters
# trainable_params = sum(1 for param in model.parameters() if param.requires_grad)
# print(f"\nTotal trainable parameter tensors: {trainable_params}")

# Verify only LoRA parameters in text_model are trainable
print("\nVerifying trainable parameters...")
cnt_trainable = 0
cnt_total = 0
for name, param in model.named_parameters():
    cnt_total += 1
    if 'lora' in name and 'text_model' in name:
        cnt_trainable += 1
        param.requires_grad = True
    else:
        param.requires_grad = False

print(f"Trainable LoRA parameter tensors in text_model: {cnt_trainable}")
print(f"Total LoRA parameter tensors: {cnt_total}")

trainable_params = sum(1 for param in model.parameters() if param.requires_grad)
print(f"\nTotal trainable parameter tensors: {trainable_params}")
print("✓ LoRA adapters configured")

# ============================================================================
# WANDB INITIALIZATION
# ============================================================================
print("\n[7/9] Initializing Weights & Biases...")
wandb.init(
    project="IE643_SmolVLM_Finetuning", 
    name="smolvlm-256M-run-language-head-finetuning-stage2-asdiv-re-2",
)
print("✓ W&B initialized")

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
print("\n[8/9] Configuring training parameters...")

training_args = SFTConfig(
    overwrite_output_dir=False,

    # Output directory for checkpoints
    output_dir="/home/ie643_mindspring/model_weights/training-language-head-finetuning-stage2-asdiv-re-2",
    
    # Training steps (total batches to process)
    num_epochs=2,
    
    # Batch size per GPU
    per_device_train_batch_size=8,
    
    # Accumulate gradients over multiple batches (effective batch size = 8 * 4 = 32)
    gradient_accumulation_steps=4,
    
    # Gradient checkpointing for memory efficiency
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # Learning rate warmup (1% of training)
    warmup_ratio=0.01,
    
    # Peak learning rate
    learning_rate=1e-5,
    
    # Weight decay for regularization
    weight_decay=0.01,
    
    # Log metrics every 20 steps
    logging_steps=20,
    
    # Checkpoint saving configuration
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,  # Keep only the last checkpoint
    
    # Optimizer
    optim="adamw_torch",
    
    # Use bfloat16 mixed precision training
    bf16=True,
    
    # Don't push to Hugging Face Hub
    push_to_hub=False,
    
    # Gradient clipping threshold
    max_grad_norm=1.0,
    
    # Report metrics to W&B
    report_to="wandb",
    
    # Maximum sequence length
    max_length=1024,
    
    # Keep all columns in dataset
    remove_unused_columns=False,
    
    # Cosine learning rate schedule with warmup
    lr_scheduler_type="cosine",
    
    # Enable training and evaluation
    do_train=True,
    do_eval=False,
)

print("✓ Training configuration complete")

# ============================================================================
# TRAINER INITIALIZATION
# ============================================================================
print("\n[9/9] Initializing trainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_train_set,
    processing_class=processor,
    data_collator=VLMCollator(processor=processor)
)

print("✓ Trainer initialized")

# ============================================================================
# TRAINING EXECUTION
# ============================================================================
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"Total steps: {training_args.max_steps}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print("=" * 60 + "\n")

# Start training (tqdm progress bars are automatically shown by Trainer)
trainer.train()

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
print(f"Model checkpoints saved to: {training_args.output_dir}")
print("=" * 60)