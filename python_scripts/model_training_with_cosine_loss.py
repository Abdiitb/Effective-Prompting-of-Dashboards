"""
Vision-Language Model (VLM) Fine-tuning Script
Fine-tunes SmolVLM-256M-Instruct on PlotQA dataset for chart interpretation
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union, Tuple
from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Idefics3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from dataclasses import dataclass
from collections import Counter
from peft import LoraConfig, get_peft_model
import numpy as np
import wandb

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
login("SECRET_HF_TOKEN")  # Replace with your actual token or use environment variable
print("✓ Successfully logged into Hugging Face")

# ============================================================================
# DATASET LOADING AND CONFIGURATION
# ============================================================================
print("\n[2/9] Loading dataset...")
dataset_id = "Abd223653/Plot-QA-V1"

# Load datasets in streaming mode for memory efficiency
train_dataset = load_dataset(dataset_id, split="train", streaming=True)

# Dataset split configuration
TRAIN_START = 0
TRAIN_END = 1000

print(f"Training samples: {TRAIN_START} to {TRAIN_END}")

# Create training subset
train_set = train_dataset.skip(TRAIN_START).take(TRAIN_END - TRAIN_START)

# Shuffle training data for better convergence
print("Shuffling training data...")
train_set = train_set.shuffle(seed=42, buffer_size=10000)
print("✓ Dataset loaded and configured")

# ============================================================================
# PROCESSOR INITIALIZATION
# ============================================================================
print("\n[3/9] Loading processor...")
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
print("✓ Processor loaded successfully")

# ============================================================================
# SYSTEM PROMPT DEFINITION
# ============================================================================
system_message = """
You are a Vision-Language Model specialized in interpreting chart and plot images.
Analyze the chart carefully and answer the given question concisely (usually a single word, number, or short phrase).
Use both visual information (values, colors, labels) and simple reasoning (e.g., finding averages, differences, trends) based on the chart data.
Do not rely on any external or prior knowledge — all answers must come from interpreting the chart itself.
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
                {"type": "image"},
                {"type": "text", "text": f"Question Template: {sample['template']}\n Plot Type: {sample['type']}\n Question: {sample['question_string']}"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["answer"]}],
        },
    ]

    # Apply chat template to convert messages to model-specific format
    text = processor.apply_chat_template(messages, tokenize=False)

    return {
        "text": text,  # Formatted conversation text
        "images": [sample["image"]]  # Associated chart image
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
        images = [ex["images"] for ex in examples]

        # Process inputs with padding and convert to tensors
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        # Create labels by copying input_ids (standard causal LM approach)
        labels = batch["input_ids"].clone()

        # Mask prompt tokens so model only learns to predict assistant responses
        for i in range(len(labels)):
            # Find newline tokens (198) to locate assistant response start
            response_start = (batch["input_ids"][i] == 198).nonzero(as_tuple=True)[0]
            if len(response_start) > 0:
                # Mask all tokens before the assistant's response with -100
                # -100 is ignored by PyTorch loss functions
                labels[i, : response_start[-2]] = -100

        batch["labels"] = labels
        return batch


class CosineSimilarityLoss(nn.Module):
    """
    Custom loss function that uses cosine similarity between embeddings of predicted output and labels.
    Properly masks ignored tokens (label = -100) following HuggingFace conventions.

    This loss is particularly useful for vision-language models where you want to match the semantic
    meaning between predicted and target sequences at the embedding level.

    Args:
        temperature (float): Temperature parameter for scaling cosine similarity. Default: 0.07
        margin (float): Margin for contrastive loss. Default: 0.0
        use_contrastive (bool): Whether to use contrastive loss variant with margin. Default: False
    """

    def __init__(self, temperature: float = 0.07, margin: float = 0.1, use_contrastive: bool = False):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.use_contrastive = use_contrastive

    def forward(
        self,
        pred_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss with proper masking of ignored tokens.

        The goal is to align the semantic meaning of predicted hidden states with target embeddings.
        This is particularly useful in VLMs where we want both:
        1. Correct token prediction (covered by LM loss)
        2. Semantically meaningful representations (covered by this cosine loss)

        Args:
            pred_embeddings: Predicted embeddings from model 
                            Shape: (batch_size, seq_len, hidden_dim) or flattened
                            These are the hidden state representations produced by the model
            target_embeddings: Target embeddings from labels 
                              Shape: (batch_size, seq_len, hidden_dim) or flattened
                              These are the embedding representations of the correct tokens
            labels: Label token IDs (batch_size, seq_len) 
                   Tokens with value -100 are ignored (padding)
                   Used for documentation/debugging purposes in this function
            attention_mask: Optional attention mask for padded positions 
                           Shape: (batch_size, seq_len)
                           1 for real tokens, 0 for padding
                           Not currently used but provided for compatibility

        Returns:
            loss: Scalar loss tensor representing mean cosine distance across all tokens
                 Loss = mean(1 - cosine_similarity) for each token pair
        """
        # Normalize embeddings to unit norm (L2 normalization)
        # This ensures cosine similarity = dot product of normalized vectors
        # Range of cosine similarity: [-1 (opposite), 0 (orthogonal), +1 (same)]
        pred_norm = F.normalize(pred_embeddings, p=2, dim=-1)    # Normalize last dimension (hidden_dim)
        target_norm = F.normalize(target_embeddings, p=2, dim=-1)

        # Compute cosine similarity: sum of element-wise products along hidden dimension
        # Broadcasting automatically handles batch and sequence dimensions
        # Result shape: (batch_size, seq_len) or (n_tokens,) if already flattened
        cos_sim = torch.sum(pred_norm * target_norm, dim=-1)

        if self.use_contrastive:
            # ===== CONTRASTIVE VARIANT =====
            # Margin-based loss: penalize if similarity falls below (1 - margin)
            # Only penalizes dissimilar pairs, ignores similar ones
            # Formula: max(0, margin - similarity)
            # This is useful for learning hard negatives
            loss_per_token = torch.clamp(self.margin - cos_sim, min=0.0)
        else:
            # ===== STANDARD COSINE DISTANCE =====
            # Simple loss: 1 - similarity
            # Minimizing this loss maximizes cosine similarity
            # Loss is 0 when embeddings are identical, increases with dissimilarity
            # This is more stable for fine-tuning
            loss_per_token = 1.0 - cos_sim

        # Average loss across all tokens
        # This produces a scalar loss value for backpropagation
        loss = loss_per_token.mean()

        return loss

class HybridLossTrainer(Trainer):
    """
    Custom Trainer that combines two complementary loss functions:
    1. Language Modeling (LM) Loss: Standard causal LM objective for next-token prediction
    2. Cosine Similarity Loss: Matches semantic meaning between predicted and target embeddings
    
    This hybrid approach helps the model learn both syntactic patterns (LM loss) and 
    semantic alignment (cosine loss), which is particularly useful for VLMs.
    
    Key Features:
    - Properly masks ignored tokens (label = -100) in both loss calculations
    - Weighted combination allows balancing between the two objectives
    - Compatible with HuggingFace Trainer ecosystem
    
    Args:
        cosine_loss_weight (float): Weight for cosine loss in final loss = (1-w)*lm + w*cosine
                                   Range [0, 1]: 0 = pure LM loss, 1 = pure cosine loss
                                   Typical range: [0.1, 0.5] for balancing
        cosine_loss_fn (CosineSimilarityLoss): Instance of custom cosine loss function
    """

    def __init__(
        self,
        *args,
        cosine_loss_weight: float = 0.6,
        cosine_loss_fn: Optional[CosineSimilarityLoss] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Weight for balancing cosine similarity loss vs LM loss
        self.cosine_loss_weight = cosine_loss_weight
        # Initialize cosine loss with default parameters if not provided
        self.cosine_loss_fn = cosine_loss_fn or CosineSimilarityLoss()
        # Standard CrossEntropyLoss with ignore_index=-100 for masked tokens
        # This is the conventional approach used in HuggingFace models
        self.lm_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute hybrid loss combining embedding cosine similarity and language modeling loss.
        Properly masks -100 tokens in both loss calculations.
        
        This method is called by the Trainer at each training step.
        It overrides the default loss computation from the base Trainer class.
        
        Args:
            model: The VLM model to compute loss for
            inputs: Dictionary containing input_ids, attention_mask, labels, pixel_values, etc.
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in the batch (for multi-GPU scenarios)
        
        Returns:
            loss: Scalar tensor representing the combined loss
            (loss, outputs): Tuple if return_outputs=True
        """
        # Extract labels and remove from inputs dict (model.forward doesn't expect them)
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Forward pass through model to get logits and hidden states
        # output_hidden_states=True enables extracting intermediate layer representations
        outputs = model(**inputs, output_hidden_states=True)

        if labels is None:
            # If no labels, use model's default loss if available
            loss = outputs.loss if hasattr(outputs, 'loss') else None
        else:
            # Extract the last hidden layer representation (before output projection)
            hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
            attention_mask = inputs.get("attention_mask", None)

            # ===== COSINE SIMILARITY LOSS =====
            # This loss tries to align predicted hidden states with label embeddings
            if labels.dim() == 2 and labels.dtype == torch.long:
                # Get the embedding layer from the model
                # Different model architectures store embeddings in different places
                if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    embedding_layer = model.model.embed_tokens
                elif hasattr(model, 'embed_tokens'):
                    embedding_layer = model.embed_tokens
                else:
                    embedding_layer = model.get_input_embeddings()

                # Create mask for non-ignored tokens (label != -100)
                # -100 are padding tokens that should not contribute to loss
                label_mask = (labels != -100)

                # Filter to keep only valid (non-ignored) tokens
                # This reduces memory usage and prevents NaN gradients from -100 indices
                filtered_labels = labels[label_mask]
                filtered_hidden_states = hidden_states[label_mask]

                # Get embedding representations for the filtered labels
                # This creates the "target" embeddings we want to match
                label_embeddings = embedding_layer(filtered_labels)
                
                # Compute cosine similarity loss between predictions and targets
                cosine_loss = self.cosine_loss_fn(
                    pred_embeddings=filtered_hidden_states,
                    target_embeddings=label_embeddings,
                    labels=labels,  # Pass for documentation/masking info
                    attention_mask=attention_mask
                )
            else:
                # Fallback for non-standard label formats
                cosine_loss = self.cosine_loss_fn(
                    pred_embeddings=hidden_states,
                    target_embeddings=labels,
                    labels=None,
                    attention_mask=attention_mask
                )

            # ===== LANGUAGE MODELING LOSS =====
            # Standard next-token prediction loss
            lm_loss = None
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                # Get model predictions (logits) from the last hidden state projection
                logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                
                # Reshape for CrossEntropyLoss: (batch_size * seq_len, vocab_size)
                # and labels: (batch_size * seq_len,)
                # CrossEntropyLoss with ignore_index=-100 automatically masks these tokens
                lm_loss = self.lm_loss(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1)
                )

            # ===== COMBINE LOSSES =====
            # Weighted combination: higher cosine_loss_weight emphasizes semantic alignment
            # Formula: (1 - w) * LM_loss + w * Cosine_loss
            if lm_loss is not None:
                loss = (1 - self.cosine_loss_weight) * lm_loss + self.cosine_loss_weight * cosine_loss
            else:
                # Fallback to cosine loss only if LM loss is unavailable
                loss = cosine_loss

        return (loss, outputs) if return_outputs else loss

# Initialize custom loss with proper settings
# This loss function will be used in the HybridLossTrainer
cosine_loss_fn = CosineSimilarityLoss(
    temperature=0.07,  # Temperature for scaling embeddings (lower = sharper distinctions)
    margin=0.1,        # Margin for contrastive loss (only used if use_contrastive=True)
    use_contrastive=False  # Set to True to enable margin-based contrastive learning
)

# ============================================================================
# MODEL LOADING WITH QUANTIZATION (OPTIONAL)
# ============================================================================
print("\n[5/9] Loading base model...")

# BitsAndBytes quantization configuration for 4-bit model compression
# This reduces VRAM usage from ~5-8GB to ~1-2GB with minimal accuracy loss
# Quantization is optional - comment out 'quantization_config' parameter if not needed
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,             # Apply quantization to quantization scale factors (nested quantization)
    bnb_4bit_quant_type="nf4",                  # Use NormalFloat4 (optimal for fine-tuning)
    bnb_4bit_compute_dtype=torch.bfloat16       # Compute in bfloat16 for numerical stability during training
)

# Load pre-trained Vision-Language Model from local directory
# SmolVLM-256M is a lightweight model suitable for resource-constrained training
model = Idefics3ForConditionalGeneration.from_pretrained(
    "/home/ie643_mindspring/models/smolvlm_256M",  # Local model path (faster, no internet needed)
    device_map={"": 0},                             # Load entire model on GPU 0
    dtype=torch.bfloat16,                           # Use bfloat16 for memory efficiency (vs float32)
    # quantization_config=bnb_config,              # UNCOMMENT to enable 4-bit quantization
    _attn_implementation="sdpa",                    # Use Scaled Dot-Product Attention (faster than default)
    local_files_only=True                           # Only load from disk, don't download from HuggingFace
)
print("✓ Base model loaded")

# ============================================================================
# LORA (PEFT) CONFIGURATION - Parameter Efficient Fine-Tuning
# ============================================================================
print("\n[6/9] Configuring LoRA adapters...")

# LoRA (Low-Rank Adaptation) adds trainable adapters to the model
# This reduces trainable parameters from millions to thousands
# Typically trades ~1-2% accuracy for 10-100× fewer trainable parameters
peft_config = LoraConfig(
    # Core LoRA hyperparameters
    r=8,                                        # LoRA rank: 8 = 256 per layer (8x8 matrices)
                                                # Higher r = more capacity but slower, more VRAM
                                                # Typical range: [4, 8, 16, 32] for fine-tuning
    lora_alpha=8,                               # Scaling factor: alpha/r controls LoRA output scale
                                                # Usually set equal to r for simplicity
    lora_dropout=0.1,                           # Dropout applied to LoRA weights (regularization)
    
    # Which transformer layers to apply LoRA to
    target_modules=[
        "down_proj",    # Feed-forward down projection (MLP layer)
        "o_proj",       # Output projection from attention head
        "k_proj",       # Key projection in multi-head attention
        "q_proj",       # Query projection in multi-head attention
        "gate_proj",    # Gated linear unit projection
        "up_proj",      # Feed-forward up projection
        "v_proj",       # Value projection in multi-head attention
        # "connector.modality_projection.proj"  # Vision-text connector (commented: test without)
    ],
    
    # Advanced LoRA variant
    use_dora=True,                              # DoRA (Weight-Decomposed LoRA): better convergence
                                                # Decomposes weights into norm and direction
                                                # Slight overhead but better fine-tuning results
    
    # Weight initialization strategy
    init_lora_weights="gaussian",               # Initialize LoRA weights from Gaussian N(0,1)
                                                # Alternative: "pissa" for PiSSA variant
)

# Apply LoRA adapters to the base model
# This converts the model into a PEFT model with trainable adapters
model = get_peft_model(model, peft_config)

# Display trainable parameters summary
print("\n" + "=" * 60)
model.print_trainable_parameters()  # Shows trainable vs total parameters ratio
print("=" * 60)

# Count and display number of trainable parameter tensors
trainable_params = sum(1 for param in model.parameters() if param.requires_grad)
print(f"\nTotal trainable parameter tensors: {trainable_params}")
print("✓ LoRA adapters configured")

# ============================================================================
# WEIGHTS & BIASES (W&B) INITIALIZATION
# ============================================================================
print("\n[7/9] Initializing Weights & Biases...")
# W&B tracks metrics, hyperparameters, and training logs for visualization
wandb.init(
    project="IE643_SmolVLM_Finetuning",  # Project name for organizing experiments
    name="smolvlm-256M-run-with-masking-without-connector-link-lora8-cosine-loss-1to1000"  # Unique run identifier
)
print("✓ W&B initialized")

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
print("\n[8/9] Configuring training parameters...")

# TrainingArguments defines all hyperparameters for the training loop
# Many of these follow best practices for fine-tuning transformer models
training_args = TrainingArguments(
    # ==== OUTPUT & CHECKPOINTING ====
    output_dir="/home/ie643_mindspring/model_weights/training-with-masking-without-connector-link-lora8-cosine-loss-1to1000",
    # Directory to save checkpoints and training artifacts
    
    # ==== TRAINING STEPS & BATCHING ====
    max_steps=250,                              # Total number of training steps (overrides epochs)
                                                # total_samples / (batch_size * accumulation_steps) = max_steps
    per_device_train_batch_size=2,              # Samples per GPU per forward pass (memory-constrained)
    gradient_accumulation_steps=2,              # Accumulate gradients over N steps before update
                                                # Effective batch size = 2 * 4 = 8
                                                # Simulates larger batches without OOM
    
    # ==== MEMORY & COMPUTATION ====
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # Enable gradient checkpointing to trade compute for memory
    # Recomputes activations during backward pass instead of storing them
    
    # ==== LEARNING RATE SCHEDULE ====
    warmup_ratio=0.05,                          # Warmup first 5% of steps with linearly increasing LR
                                                # Helps stabilize training in early stages
    learning_rate=5e-5,                         # Peak learning rate (after warmup)
                                                # Typical range for fine-tuning: [1e-5, 5e-4]
    weight_decay=0.01,                          # L2 regularization coefficient (for AdamW)
                                                # Prevents overfitting by penalizing large weights
    
    # ==== LOGGING & MONITORING ====
    logging_steps=75,                           # Log metrics every 75 steps
    report_to="wandb",                          # Send metrics to Weights & Biases
    
    # ==== CHECKPOINT MANAGEMENT ====
    save_strategy="steps",                      # Save checkpoint at specified intervals
    save_steps=75,                              # Save after every 75 steps (log_steps)
    save_total_limit=1,                         # Keep only 1 checkpoint (save disk space)
                                                # Only latest checkpoint retained
    
    # ==== OPTIMIZER & PRECISION ====
    optim="adamw_torch",                        # Use PyTorch's AdamW (no external dependencies)
                                                # Good general-purpose optimizer for fine-tuning
    bf16=True,                                  # Use bfloat16 mixed precision training
                                                # Reduces memory by ~50%, speeds up training
                                                # Negligible accuracy loss compared to fp32
    
    # ==== GRADIENT & STABILITY ====
    max_grad_norm=1.0,                          # Gradient clipping threshold
                                                # Prevents exploding gradients in transformers
    
    # ==== MISC ====
    push_to_hub=False,                          # Don't upload final model to HuggingFace Hub
    
    remove_unused_columns=False,                # Keep all columns (needed for custom collator)
    lr_scheduler_type="cosine",                 # Cosine annealing: smooth LR decay with warm restarts
                                                # Alternative: "linear", "polynomial", "constant"
    do_train=True,                              # Enable training
    do_eval=False,                              # Disable evaluation (no validation set)
    
    # Uncomment to enable evaluation
    # eval_strategy="steps",                    # Evaluate at specified intervals
    # eval_steps=75,                            # Evaluate every 75 steps
)

print("✓ Training configuration complete")

# ============================================================================
# TRAINER INITIALIZATION
# ============================================================================
print("\n[9/9] Initializing trainer...")

# Initialize the custom data collator that handles batching of images and text
vlm_collator = VLMCollator(processor=processor)

# Create HybridLossTrainer instance
# This combines cosine similarity loss with standard language modeling loss
trainer = HybridLossTrainer(
    model=model,                                # The model to train (with LoRA adapters)
    args=training_args,                         # Training hyperparameters
    train_dataset=formatted_train_set,          # Training data (1000 samples)
    cosine_loss_fn=cosine_loss_fn,              # Custom cosine loss function
    data_collator=vlm_collator,                 # Custom batching logic
    processing_class=processor,                 # Processor for tokenization/image processing
    cosine_loss_weight=0.1,                     # Weight for cosine loss in hybrid objective
                                                # 0.1 = 90% LM loss + 10% cosine loss
    # callbacks=[LoRAGradientCheckCallback()],  # Optional: monitor LoRA gradient flow
)

print("✓ Trainer initialized")

# ============================================================================
# TRAINING EXECUTION
# ============================================================================
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

# Display training configuration summary
print(f"Total steps           : {training_args.max_steps}")
print(f"Batch size            : {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation : {training_args.gradient_accumulation_steps}")
print(f"Effective batch size  : {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print("=" * 60 + "\n")

# Start the training loop
# The Trainer handles:
# - Distributed training (if multiple GPUs)
# - Mixed precision training (bf16)
# - Gradient accumulation
# - Checkpointing and resumption
# - Metrics logging to W&B
# - Progress bar visualization via tqdm
trainer.train()

# ============================================================================
# POST-TRAINING SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
print(f"Model checkpoints saved to: {training_args.output_dir}")
print("✓ Training finished successfully")
print("=" * 60)
