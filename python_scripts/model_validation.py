# ===============================================
# 🧠 VALIDATION SCRIPT FOR SmolVLM-256M MODEL
# ===============================================
# Purpose: Validate fine-tuned SmolVLM-256M model on test dataset
# Output: CSV predictions and evaluation metrics by question type

import os
import gc
import logging
import time
import torch
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
from PIL import Image
from io import BytesIO

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# STAGE 1: ENVIRONMENT AND DEVICE SETUP
# ============================================================================
logger.info("=" * 60)
logger.info("STAGE 1: ENVIRONMENT & DEVICE SETUP")
logger.info("=" * 60)

# Configure CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"CUDA Available: {torch.cuda.is_available()}")
logger.info(f"Device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU Name: {torch.cuda.get_device_name(device)}")
    gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    logger.info(f"GPU Memory: {gpu_memory_gb:.2f} GB")

# Authenticate with Hugging Face Hub
logger.info("Authenticating with Hugging Face Hub...")
login("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")
logger.info("✓ Successfully logged into Hugging Face")
logger.info("=" * 60)

# -----------------------------------------------
# 📂 2. Load Dataset (Streaming for Efficiency)
# -----------------------------------------------
logger.info("\n" + "=" * 60)
logger.info("STAGE 2: DATASET LOADING")
logger.info("=" * 60)

dataset_id = "Abd223653/SmolVLM_Val_Data"
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

logger.info(f"\nDataset ID: {dataset_id}")
logger.info(f"Model ID: {model_id}")

# Load processor for vision-language model (handles image + text inputs)
logger.info("\nLoading processor...")
processor = AutoProcessor.from_pretrained(model_id)

# Fix padding for decoder-only architecture
# Set padding_side to 'left' for correct generation with batched inputs
if hasattr(processor, "tokenizer"):
    processor.tokenizer.padding_side = "left"
    logger.info(f"Tokenizer padding_side set to: {processor.tokenizer.padding_side}")
else:
    logger.warning("Could not set tokenizer padding_side")

logger.info("✓ Processor loaded successfully")

# Load validation data
logger.info("\nLoading validation dataset...")
val_set = load_dataset(dataset_id, split="arithmetic")
logger.info("✓ Validation dataset loaded")
logger.info("=" * 60)

# 🧾 3. Define System Message and Data Formatting
# -----------------------------------------------
system_message = """
You are an assistant that solves numerical and plot-based questions.
Perform calculations when needed, show clear step-by-step reasoning, and provide the final answer after a blank line in the format: ####<answer>.
For multi-step problems, show all steps.
For single-step problems, give the final answer with ####.
Respect parentheses and operation precedence.
"""


def format_data_val(sample):
    """
    Formats raw dataset samples into model-compatible input format.

    Args:
        sample (dict): A single dataset sample with keys like 'image', 'question_string', and 'answer'.

    Returns:
        dict: Contains formatted 'text', 'images', and metadata (question, answer, image_index).
    """
    image = None

    if sample["image"] is not None:
        # Convert bytes to a file-like object
        image_stream = BytesIO(sample["image"]["bytes"])

        # Open the image with Pillow
        image = Image.open(image_stream)

        # Construct multi-turn conversation with system, user, and assistant messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{sample['question']}"},
                ],
            },
        ]

    else:
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
        ]

    # Apply chat template to convert messages to model-specific format
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return {
        "text": text,  # Formatted conversation text
        "images": image,  # Associated chart image
        "question": sample["question"],  # Preserve original question
        "answer": sample["answer"],  # Preserve true answer
        "type": sample["type"],  # Preserve question type
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
        raw_images = [ex["images"] for ex in examples]
        # images_for_processor = [[img] if img is not None else [] for img in raw_images]

        # Tokenize & preprocess with left-padding for decoder-only models
        batch = self.processor(
            text=texts,
            images=raw_images,
            return_tensors="pt",
            padding=True,
            padding_side="left",  # Required for decoder-only architecture
        )

        # Preserve metadata (question, answer, image_index)
        batch["questions"] = [ex["question"] for ex in examples]
        batch["answers"] = [ex["answer"] for ex in examples]
        batch["images"] = raw_images
        batch["types"] = [ex["type"] for ex in examples]

        return batch


logger.info("\n" + "=" * 60)
logger.info("STAGE 4: DATA COLLATOR INITIALIZATION")
logger.info("=" * 60)
logger.info("✓ VLMCollator class defined and ready")
logger.info("=" * 60)

# ============================================================================
# STAGE 5: MODEL LOADING AND QUANTIZATION
# ============================================================================
# Enable 4-bit quantization for reduced VRAM usage (inference-friendly)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

logger.info("\n" + "=" * 60)
logger.info("STAGE 5: MODEL LOADING & QUANTIZATION")
logger.info("=" * 60)
logger.info("\nQuantization Config:")
logger.info("✓ 4-bit quantization enabled")
logger.info("✓ Double quantization enabled")
logger.info("✓ Quantization type: nf4")
logger.info("✓ Compute dtype: bfloat16")

# Load pretrained base model with quantization
logger.info("\nLoading pretrained base model...")
model = Idefics3ForConditionalGeneration.from_pretrained(
    "/home/ie643_mindspring/models/smolvlm_256M",
    device_map={"": 0},
    dtype=torch.bfloat16,
    # quantization_config=bnb_config,  # Enable 4-bit quantization
    _attn_implementation="sdpa",
    local_files_only=True,
)
logger.info("✓ Base model loaded successfully")

# Log model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"\nModel Architecture:")
logger.info(f"✓ Total Parameters: {total_params:,}")
logger.info(f"✓ Trainable Parameters: {trainable_params:,}")

# Load LoRA adapter checkpoint for transfer learning
logger.info("\nLoading LoRA adapter checkpoint...")
adapter_path = (
    "/home/ie643_mindspring/model_weights/training_stage2_image_text_part3/checkpoint-494"
)
model.load_adapter(adapter_path)
logger.info(f"✓ LoRA adapter loaded from: {adapter_path}")

model.eval()  # Set to evaluation mode (disables dropout, etc.)
logger.info("✓ Model set to evaluation mode")
logger.info("=" * 60)

# ============================================================================
# STAGE 6: RUN INFERENCE ON VALIDATION SET (STREAMING CSV SAVE)
# ============================================================================
# This section streams predictions directly to CSV to avoid memory overflow
import csv

preds = []
questions = []
answers = []
image_indexes = []

logger.info("\n" + "=" * 60)
logger.info("STAGE 6: INFERENCE ON VALIDATION SET (STREAMING OUTPUT)")
logger.info("=" * 60)

batch_size = 16
batch_items = []
collator = VLMCollator(processor=processor)

output_path = "/home/ie643_mindspring/results/csv/model_validation_streamed_training_stage2_image_text_part3.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Prepare CSV and write header once
if not os.path.exists(output_path):
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # CSV columns: image filename, question text, ground truth answer, model prediction, question type
        writer.writerow(["image", "question", "true_answer", "model_answer", "type"])

logger.info(f"\nInference Configuration:")
logger.info(f"✓ Batch size: {batch_size}")
logger.info(f"✓ Output file: {output_path}")
logger.info(f"✓ Writing results incrementally (stream mode)\n")

start_time = time.time()
batch_count = 0
processed_samples = 0

# Main inference loop: process samples in batches
for item in tqdm(formatted_val_set, desc="Generating Responses"):
    batch_items.append(item)

    # Process batch when it reaches batch_size
    if len(batch_items) == batch_size:
        # Collate batch items into tensors
        batch = collator(batch_items)
        batch_questions = batch.pop("questions")
        batch_answers = batch.pop("answers")
        batch_images = batch.pop("images")
        batch_types = batch.pop("types")

        # Move tensors to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Generate predictions for entire batch
        with torch.no_grad():
            generated_ids = model.generate(
                **batch,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=True,
            )

        generated_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Write results of this batch to CSV
        with open(output_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for i, full_text in enumerate(generated_texts):
                assistant_idx = full_text.rfind("Assistant: ")
                prediction = (
                    full_text[assistant_idx + len("Assistant: ") :].strip()
                    if assistant_idx != -1
                    else ""
                )
                writer.writerow(
                    [
                        batch_images[i],
                        batch_questions[i],
                        batch_answers[i],
                        prediction,
                        batch_types[i],
                    ]
                )

        processed_samples += len(batch_items)
        batch_items = []
        batch_count += 1

        # Cleanup to free GPU memory between batches
        del batch, generated_ids, generated_texts
        gc.collect()
        torch.cuda.empty_cache()

        # Log progress every 10 batches
        if batch_count % 10 == 0:
            logger.info(f"✓ Processed {processed_samples:,} samples so far...")

# Handle remaining items
if batch_items:
    batch = collator(batch_items)
    batch_questions = batch.pop("questions")
    batch_answers = batch.pop("answers")
    batch_images = batch.pop("images")

    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

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
            prediction = (
                full_text[assistant_idx + len("Assistant: ") :].strip()
                if assistant_idx != -1
                else ""
            )
            writer.writerow(
                [
                    batch_images[i],
                    batch_questions[i],
                    batch_answers[i],
                    prediction,
                    batch_types[i],
                ]
            )

    processed_samples += len(batch_items)
    batch_count += 1

elapsed_time = time.time() - start_time
logger.info("-" * 60)
logger.info(f"✓ Inference completed successfully!")
logger.info(f"✓ Total processed samples: {processed_samples:,}")
logger.info(f"✓ Total batches: {batch_count}")
logger.info(
    f"✓ Total time: {elapsed_time:.2f}s ({elapsed_time / processed_samples:.2f}s/sample)"
)
logger.info(f"✓ Results saved incrementally at: {output_path}")
logger.info("=" * 60)