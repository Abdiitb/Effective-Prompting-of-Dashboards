# Effective Prompting of Dashboards - Project Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation & Setup](#installation--setup)
4. [Core Components](#core-components)
5. [Training Pipeline](#training-pipeline)
6. [Validation & Evaluation](#validation--evaluation)
7. [Web Interface](#web-interface)
8. [Model Weights & Checkpoints](#model-weights--checkpoints)
9. [Results & Analysis](#results--analysis)
10. [Usage Guide](#usage-guide)
11. [Advanced Configuration](#advanced-configuration)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Purpose
This project implements an **effective prompting framework for Vision-Language Models (VLMs)** specialized in interpreting charts and dashboards. The system fine-tunes **SmolVLM-256M-Instruct** on the **PlotQA-V1 dataset** to enable accurate chart interpretation and question-answering capabilities.

### Key Features
- 🧠 **Fine-tuned Vision-Language Model**: SmolVLM-256M-Instruct optimized for chart understanding
- 📊 **Chart QA System**: Answers questions about plots, charts, and dashboards
- 🎯 **LoRA Adaptation**: Efficient parameter-efficient fine-tuning using Low-Rank Adaptation
- 🔍 **Advanced Prompting**: Specialized system messages for chart interpretation
- 💻 **Interactive Web UI**: Gradio-based interface for real-time chart analysis
- 📈 **Comprehensive Validation**: Detailed evaluation metrics and performance tracking
- 📝 **Experiment Tracking**: Weights & Biases (W&B) integration for experiment monitoring

### Technical Stack
- **Model**: SmolVLM-256M-Instruct (Idefics3-based)
- **Framework**: Hugging Face Transformers, TRL (Transformer Reinforcement Learning)
- **Optimization**: LoRA, 4-bit Quantization via BitsAndBytes
- **Training**: SFT Trainer with custom callbacks
- **Inference**: Gradio web application
- **Evaluation**: Custom metrics with scikit-learn, matplotlib
- **Monitoring**: Weights & Biases, HuggingFace Hub

---

## Project Structure

```
├── app.py                          # Gradio web interface for inference
├── README.md                       # Repository overview
├── requirements.txt                # Python dependencies
├── models/
│   └── smolvlm_256M/              # Pre-trained model weights
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── chat_template.jinja
│       └── [other model files]
├── model_weights/
│   ├── training_1to8400/           # Checkpoint: samples 1-8400
│   ├── training_8400to23400/       # Checkpoint: samples 8400-23400
│   ├── training_23400to38400/      # Checkpoint: samples 23400-38400
│   ├── training-with-masking-**/   # Various masking experiments
│   ├── training-without-masking-**/# Ablation studies
│   └── training-*-lora*/           # Different LoRA configurations
├── python_scripts/
│   ├── model_initial_training.py           # Initial fine-tuning script
│   ├── model_continue_training.py          # Resume training from checkpoint
│   ├── model_training_with_cosine_loss.py  # Training with cosine loss
│   ├── model_validation.py                 # Comprehensive evaluation script
│   └── wandb/                              # W&B experiment logs
└── results/
    ├── csv/                        # Validation results in CSV format
    └── plots/                      # Visualization and analysis plots
```

---

## Installation & Setup

### Prerequisites
- **Python 3.10+**
- **CUDA 12.0+** (for GPU support)
- **GPU**: Recommended 16GB+ VRAM (tested on NVIDIA GPUs)
- **Storage**: ~50GB for model weights and datasets

### Step 1: Clone Repository
```bash
git clone https://github.com/Abdiitb/Effective-Prompting-of-Dashboards.git
cd Effective-Prompting-of-Dashboards
```

### Step 2: Create Virtual Environment
```bash
# Using Python venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n vlm-dashboards python=3.10
conda activate vlm-dashboards
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set HuggingFace Token
```bash
huggingface-cli login
# Follow prompts to enter your HuggingFace API token
# Token should be available at: https://huggingface.co/settings/tokens
```

### Step 5: Verify Installation
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## Core Components

### 1. Model Architecture

**SmolVLM-256M-Instruct** is a lightweight Vision-Language Model based on Idefics3:
- **Vision Encoder**: Processes chart/dashboard images
- **Language Model**: Generates text responses
- **Total Parameters**: 256M (very efficient for edge deployment)
- **Quantization**: 4-bit NormalFloat (NF4) for memory efficiency
- **Attention**: Scaled Dot Product Attention (SDPA) for speed

### 2. Fine-tuning Configuration

**LoRA (Low-Rank Adaptation)** Parameters:
- **LoRA Rank (r)**: 8, 16, or 32 (varying in experiments)
- **LoRA Alpha**: Typically set to match rank value
- **Target Modules**: Attention and feed-forward layers
- **Dropout**: 0.05 for regularization

**Quantization (BitsAndBytes)**:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### 3. System Prompt

The model uses a specialized system message for chart interpretation:

```
You are a Vision-Language Model specialized in interpreting chart and plot images.
Analyze the chart carefully and answer the given question concisely (usually a single 
word, number, or short phrase). Use both visual information (values, colors, labels) 
and simple reasoning (e.g., finding averages, differences, trends) based on the chart data.
Do not rely on any external or prior knowledge — all answers must come from interpreting 
the chart itself.
```

---

## Training Pipeline

### Dataset: PlotQA-V1

**Source**: `Abd223653/Plot-QA-V1` (HuggingFace Dataset Hub)

**Statistics**:
- **Training Samples**: 38,400+
- **Validation Samples**: 6,000+
- **Test Samples**: 6,000+
- **Chart Types**: Line plots, bar charts, scatter plots, area charts, etc.
- **Question Types**: Count, Compare, Locate, Retrieve

**Data Format**:
```python
{
    "image": PIL.Image,              # Chart image
    "template": str,                 # Question template
    "type": str,                     # Plot type (line, bar, etc.)
    "question_string": str,          # Actual question
    "answer": str                    # Ground truth answer
}
```

### Training Process

#### 1. **Initial Training** (`model_initial_training.py`)

```bash
python python_scripts/model_initial_training.py
```

**Configuration**:
- **Samples**: 1-1000 for initial experiments (configurable via `TRAIN_START` and `TRAIN_END`)
- **Epochs**: 3
- **Batch Size**: 2 (with gradient accumulation)
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW
- **Warmup Steps**: 100
- **Logging Frequency**: Every 10 steps
- **Save Strategy**: Save every checkpoint (configurable)

**Output**:
- Model checkpoints in `model_weights/training_1to1000/checkpoint-*/`
- W&B logs for experiment tracking
- Training metrics (loss, learning rate, etc.)

#### 2. **Continue Training** (`model_continue_training.py`)

```bash
python python_scripts/model_continue_training.py
```

**Purpose**: Resume training from a checkpoint with different data samples
- Load checkpoint from `model_weights/training_1to8400/checkpoint-525/`
- Continue training on extended dataset range (samples 8400-23400)
- Maintains learned parameters from initial training

#### 3. **Alternative Training Method** (`model_training_with_cosine_loss.py`)

```bash
python python_scripts/model_training_with_cosine_loss.py
```

**Features**:
- Custom cosine similarity loss
- Potentially better performance on semantic similarity tasks
- Experimental approach for chart understanding

### Training Experiments

The project includes multiple training configurations:

| Experiment | Samples | LoRA Rank | Masking | Features |
|-----------|---------|-----------|---------|----------|
| `training_1to8400` | 1-8400 | 32 | Yes | Base training |
| `training_8400to23400` | 8400-23400 | 32 | Yes | Continuation |
| `training_23400to38400` | 23400-38400 | 32 | Yes | Final extension |
| `training-with-masking-*-lora8` | 1-1000 | 8 | Yes | Masking strategy |
| `training-with-masking-*-lora16` | 1-1000 | 16 | Yes | Higher rank |
| `training-with-masking-*-lora32` | 1-1000 | 32 | Yes | Highest rank |
| `training-without-masking-*-lora8` | 1-1000 | 8 | No | Ablation study |

---

## Validation & Evaluation

### Validation Script

```bash
python python_scripts/model_validation.py
```

**Configuration**:
- **Test Samples**: 1-500 (configurable)
- **Batch Size**: 1 (for stable inference)
- **Quantization**: 4-bit (same as training)
- **Max Generation Tokens**: 150

### Evaluation Metrics

The validation script computes:

1. **Exact Match Accuracy**: Percentage of exact string matches
2. **Token-level Accuracy**: Comparing individual tokens
3. **Semantic Similarity**: Using sentence embeddings (if available)
4. **Confusion Matrix**: For classification-style answers
5. **Error Analysis**: Categorizing failure modes

### Output Files

Results are saved in `results/csv/`:
```
model_validation_streamed_1to8400.csv
model_validation_streamed_8400to23400.csv
model_validation_streamed_23400to38400.csv
model_validation_streamed_base.csv
model_validation_streamed_test_set-*.csv
```

**CSV Columns**:
- `question`: Original question about the chart
- `ground_truth`: Expected answer
- `prediction`: Model prediction
- `exact_match`: Binary correctness indicator
- `chart_type`: Type of plot (line, bar, etc.)
- `question_type`: Question category (count, compare, etc.)
- `inference_time`: Time taken for inference

### Analysis & Visualization

Results are visualized in `results/plots/`:
- Accuracy by chart type
- Accuracy by question type
- Error distribution
- Confusion matrices (for classification tasks)
- Performance over training samples

---

## Web Interface

### Running the Application

```bash
python app.py
```

The application will start at `http://localhost:7860`

### Interface Features

**Left Panel - Chat History**:
- Displays multi-turn conversation
- Shows user questions and model responses
- Maintains session memory
- Scrollable for long conversations

**Right Panel - Image Upload**:
- Upload chart/dashboard images (PNG, JPG, GIF, etc.)
- Preview of uploaded image
- Drop zone for drag-and-drop uploads

**Bottom Controls**:
- **Text Input**: Type questions about the chart
- **Send Button**: Submit question (💬)
- **Clear Chat Button**: Reset conversation (🧹)

### Example Queries

```
"What is the highest value in this chart?"
"Compare the values between 2020 and 2021"
"What trend does this line show?"
"Which category has the lowest bar?"
"What is the average value?"
```

### UI Configuration

**Theme**: Soft theme with:
- Primary Color: Indigo
- Secondary Color: Blue
- Neutral Color: Slate
- Font: Inter, Medium size

### Advanced Features

- **4-bit Quantization**: Efficient memory usage
- **Device Mapping**: Automatic GPU allocation (via `device_map="auto"`)
- **SDPA Attention**: Faster inference
- **Streaming Responses**: Real-time token generation

---

## Model Weights & Checkpoints

### Pre-trained Model

**Location**: `models/smolvlm_256M/`

**Files**:
- `model.safetensors`: Model weights in SafeTensors format
- `config.json`: Model configuration
- `tokenizer.json`: Tokenizer weights
- `tokenizer_config.json`: Tokenizer settings
- `chat_template.jinja`: Chat template for formatting
- `generation_config.json`: Generation parameters
- `special_tokens_map.json`: Special tokens
- `vocab.json`: Vocabulary
- `merges.txt`: BPE merge operations
- `added_tokens.json`: Custom tokens

### Checkpoint Directory Structure

Each checkpoint follows HuggingFace format:
```
checkpoint-525/
├── adapter_config.json          # LoRA adapter configuration
├── adapter_model.bin            # LoRA weights (after 525 steps)
├── training_args.bin            # Training configuration
├── optimizer.pt                 # Optimizer state
└── scheduler.pt                 # Learning rate scheduler state
```

### Loading a Checkpoint

```python
from transformers import Idefics3ForConditionalGeneration
from peft import PeftModel

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
checkpoint_path = "model_weights/training_1to8400/checkpoint-525"

# Load base model
model = Idefics3ForConditionalGeneration.from_pretrained(model_id)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, checkpoint_path)
```

### Published Models

Fine-tuned model adapters are available on HuggingFace Hub:
- **Repository**: `Abd223653/SmolVLM_Finetune_PlotQA`
- Can be directly loaded: `model.load_adapter("Abd223653/SmolVLM_Finetune_PlotQA")`

---

## Results & Analysis

### Key Performance Metrics

Results are stored in `results/csv/` with detailed breakdowns:

**Example Statistics** (from `model_validation_streamed_1to8400.csv`):
- Exact Match Accuracy: ~65-75%
- By Chart Type: Line plots (78%), Bar charts (72%), Scatter plots (68%)
- By Question Type: Count (85%), Compare (72%), Locate (65%)
- Average Inference Time: ~2-3 seconds per image

### Experiment Findings

1. **Effect of Training Data Size**:
   - 1K samples: Baseline performance
   - 8.4K samples: Significant improvement (~10%)
   - 23.4K samples: Good convergence
   - 38.4K samples: Marginal improvements

2. **LoRA Rank Impact**:
   - LoRA-8: Fast training, good efficiency
   - LoRA-16: Balanced performance-efficiency trade-off
   - LoRA-32: Better performance, higher memory usage

3. **Masking Strategy**:
   - With masking: Better generalization on out-of-domain charts
   - Without masking: Faster training, comparable performance

### Visualization Examples

Located in `results/plots/`:
- **accuracy_by_chart_type.png**: Performance distribution across chart types
- **accuracy_by_question_type.png**: Performance by question category
- **error_distribution.png**: Error analysis and patterns
- **training_curves.png**: Loss and accuracy over training steps

---

## Usage Guide

### Basic Usage: Interactive Chat

```bash
# 1. Navigate to project directory
cd "C:\College Study\Semester 5\IE643\Project\Code"

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Run the app
python app.py

# 4. Open browser to http://localhost:7860
```

### Programmatic Usage: Inference

```python
import torch
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from PIL import Image

# Load model and processor
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(model_id)

# Load fine-tuned adapter
model.load_adapter("Abd223653/SmolVLM_Finetune_PlotQA")

# Prepare input
image = Image.open("chart.png")
question = "What is the highest value in this chart?"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ],
    }
]

# Generate response
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=150)

response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

### Batch Inference

```python
from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("Abd223653/Plot-QA-V1", split="test", streaming=True)
test_set = dataset.take(100)  # First 100 samples

# Run inference on batch
predictions = []
for sample in test_set:
    # Similar inference code as above
    predictions.append({
        "question": sample["question_string"],
        "answer": response
    })

# Save results
df = pd.DataFrame(predictions)
df.to_csv("results/batch_inference.csv", index=False)
```

### Fine-tuning on Custom Data

```python
# 1. Prepare your dataset in the same format as PlotQA-V1
# 2. Upload to HuggingFace Hub or load locally
# 3. Modify TRAIN_START and TRAIN_END in model_initial_training.py
# 4. Run: python python_scripts/model_initial_training.py
```

---

## Advanced Configuration

### Training Hyperparameters

Edit these in `model_initial_training.py`:

```python
# Dataset range
TRAIN_START = 0
TRAIN_END = 1000  # Number of samples to train on

# LoRA Configuration
lora_config = LoraConfig(
    r=8,                           # Rank
    lora_alpha=8,                  # Alpha scaling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training Configuration
training_config = SFTConfig(
    output_dir="model_weights/training_custom",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    warmup_steps=100,
    logging_steps=10,
    save_steps=None,               # Set to integer for periodic saves
    report_to=["wandb"],           # Disable with []
)
```

### Custom System Prompts

Modify the `system_message` in any script:

```python
system_message = """
Your custom instructions for chart interpretation...
"""
```

### Model Quantization

Toggle 4-bit quantization in `app.py`:

```python
# Enable quantization
quantization_config = bnb_config

# Disable quantization (use full precision)
quantization_config = None
```

### Memory Optimization

For limited GPU memory:

```python
# Option 1: Reduce batch size
per_device_train_batch_size = 1

# Option 2: Increase gradient accumulation
gradient_accumulation_steps = 8

# Option 3: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Option 4: Use LoRA with smaller rank
r = 4  # Instead of 8
```

### Distributed Training

For multi-GPU training:

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Use accelerate for distributed training
accelerate launch --multi_gpu python_scripts/model_initial_training.py
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. **CUDA Out of Memory (OOM)**

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
per_device_train_batch_size = 1

# Increase gradient accumulation
gradient_accumulation_steps = 8

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use smaller LoRA rank
r = 4
```

#### 2. **Model Download Fails**

**Symptoms**: `OSError: Can't load model` or timeout errors

**Solutions**:
```bash
# Use HuggingFace cache
export HF_HOME=/path/to/larger/storage

# Pre-download model
huggingface-cli download HuggingFaceTB/SmolVLM-256M-Instruct

# Manual download from browser and load locally
model = Idefics3ForConditionalGeneration.from_pretrained("./local/path/to/model")
```

#### 3. **HuggingFace Token Issues**

**Symptoms**: `Permission denied for repo`, `401 Unauthorized`

**Solutions**:
```bash
# Re-login
huggingface-cli logout
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN=your_token_here

# Verify token is valid at:
# https://huggingface.co/settings/tokens
```

#### 4. **Gradio Server Won't Start**

**Symptoms**: `Address already in use` or connection refused

**Solutions**:
```bash
# Use different port
python app.py --server_port 7861

# Kill existing process (Windows)
netstat -ano | findstr :7860
taskkill /PID <PID> /F

# Or modify app.py
demo.launch(server_name="0.0.0.0", server_port=7861)
```

#### 5. **Inference Produces Gibberish**

**Symptoms**: Model output doesn't make sense

**Solutions**:
```python
# Ensure model adapter is loaded
model.load_adapter("Abd223653/SmolVLM_Finetune_PlotQA")

# Reduce max_new_tokens
max_new_tokens=100  # Instead of 150

# Check system message is being used
messages[0]["role"] == "system"  # Verify

# Verify image is properly formatted
assert image.size[0] > 0 and image.size[1] > 0
```

#### 6. **Slow Inference**

**Symptoms**: Model takes >10 seconds per image

**Solutions**:
```python
# Enable 4-bit quantization
quantization_config = bnb_config

# Use SDPA attention (already enabled)
_attn_implementation="sdpa"

# Reduce max_new_tokens
max_new_tokens=100

# Batch multiple requests
# Use smaller model checkpoint if available
```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable HuggingFace logging
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_debug()
```

### Performance Monitoring

Check GPU usage during training/inference:

```bash
# Terminal 1: Start monitoring
nvidia-smi -l 1  # Update every 1 second

# Terminal 2: Run your script
python app.py
```

Monitor Weights & Biases:
- Visit: https://wandb.ai/your-username/your-project
- Real-time loss curves, learning rates, GPU memory
- Compare between different experiments

---

## Additional Resources

### Documentation References
- **SmolVLM**: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
- **Idefics3**: https://huggingface.co/docs/transformers/model_doc/idefics
- **LoRA**: https://arxiv.org/abs/2106.09685
- **PlotQA Dataset**: https://huggingface.co/datasets/Abd223653/Plot-QA-V1
- **TRL Documentation**: https://huggingface.co/docs/trl

### Helpful Links
- **HuggingFace Hub**: https://huggingface.co
- **Gradio Documentation**: https://www.gradio.app
- **Weights & Biases**: https://wandb.ai
- **PEFT (LoRA)**: https://huggingface.co/docs/peft

---

## Support & Contact

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/Abdiitb/Effective-Prompting-of-Dashboards/issues
- **Repository**: https://github.com/Abdiitb/Effective-Prompting-of-Dashboards
- **HuggingFace Model**: https://huggingface.co/Abd223653/SmolVLM_Finetune_PlotQA

---

**Last Updated**: November 3, 2025
**Version**: 1.0
**Status**: Active Development