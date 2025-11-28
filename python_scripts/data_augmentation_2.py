"""
Data Augmentation Pipeline for Plot-QA Dataset

Streams the original Plot-QA dataset, augments answers using SmolLM2,
and uploads augmented data to Hugging Face in sharded Arrow format.

Pipeline stages:
1. Environment & device setup (GPU configuration)
2. Hugging Face authentication
3. Configuration loading
4. Dataset loading in streaming mode
5. Model and tokenizer loading
6. Data augmentation and streaming upload
7. Final shard handling
8. Summary and completion
"""

import json
import logging
import os
import time
from io import BytesIO
from typing import Optional, Dict, Any

import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, login
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CHECKPOINT MANAGER FOR FAILURE RECOVERY
# ============================================================================
class CheckpointManager:
    """
    Manages pipeline checkpoints to enable resume after network failures.
    Saves progress periodically and allows resuming from last checkpoint.
    """
    
    def __init__(self, checkpoint_file: str = "augmentation_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint from file if it exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded checkpoint: {data}")
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return {"shard_id": 0, "rows_processed": 0, "last_upload": None}
    
    def save(self, shard_id: int, rows_processed: int, last_uploaded_shard: Optional[int] = None) -> None:
        """Save current progress to checkpoint file."""
        self.checkpoint_data = {
            "shard_id": shard_id,
            "rows_processed": rows_processed,
            "last_upload": last_uploaded_shard,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=2)
            logger.debug(f"Checkpoint saved: {self.checkpoint_file}")
        except IOError as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def get_resume_shard_id(self) -> int:
        """Get shard_id to resume from."""
        return self.checkpoint_data.get("shard_id", 0)
    
    def get_rows_processed(self) -> int:
        """Get total rows already processed."""
        return self.checkpoint_data.get("rows_processed", 0)
    
    def get_last_upload(self) -> Optional[int]:
        """Get last successfully uploaded shard ID."""
        return self.checkpoint_data.get("last_upload")
    
    def clear(self) -> None:
        """Clear checkpoint after successful completion."""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logger.info("Checkpoint cleared after successful completion")

# ============================================================================
# STAGE 1: ENVIRONMENT AND DEVICE SETUP
# ============================================================================
print("\n" + "=" * 70)
print("STAGE 1: ENVIRONMENT & DEVICE SETUP")
print("=" * 70)

logger.info("Setting up CUDA device configuration...")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✓ CUDA Available        : {torch.cuda.is_available()}")
print(f"✓ Device               : {device}")
if torch.cuda.is_available():
    print(f"✓ GPU Name             : {torch.cuda.get_device_name(device)}")
    gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"✓ GPU Memory           : {gpu_memory_gb:.2f} GB")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(device)} ({gpu_memory_gb:.2f} GB)")
else:
    logger.warning("CUDA not available, using CPU (slower)")

print("=" * 70 + "\n")

# ============================================================================
# STAGE 2: HUGGING FACE AUTHENTICATION
# ============================================================================
print("=" * 70)
print("STAGE 2: HUGGING FACE AUTHENTICATION")
print("=" * 70)

logger.info("Authenticating with Hugging Face Hub...")
login("YOUR_HF_TOKEN")
logger.info("✓ Successfully logged into Hugging Face Hub")

print("✓ Successfully logged into Hugging Face")
print("=" * 70 + "\n")

# ============================================================================
# STAGE 3: CONFIGURATION
# ============================================================================
logger.info("Loading configuration...")

# Dataset configuration
SOURCE_REPO = "Abd223653/SmolVLM_Training_Data_Part_3"
TARGET_REPO = "Abd223653/PlotQA_Augmented"
SPLIT = "train_3"
ROWS_PER_SHARD = 2000

# Sample limiting configuration
# Set to None to process entire dataset, or set to a number (e.g., 1000) to limit
MAX_SAMPLES = 14800  # Change to limit (e.g., 1000 for testing)

# Model configuration
CHECKPOINT = "meta-llama/Llama-3.2-3B-Instruct"
MAX_NEW_TOKENS = 128

# Batch processing configuration
BATCH_SIZE = 8  # Number of samples to process simultaneously
WRITE_BATCH_SIZE = 32  # Number of augmented rows to accumulate before writing to Arrow

logger.info(f"Source repository: {SOURCE_REPO}")
logger.info(f"Target repository: {TARGET_REPO}")
logger.info(f"Split: {SPLIT}")
logger.info(f"Rows per shard: {ROWS_PER_SHARD}")
if MAX_SAMPLES is not None:
    logger.warning(f"SAMPLE LIMIT ENABLED: Processing only {MAX_SAMPLES:,} samples")
else:
    logger.info("Processing entire dataset (no sample limit)")
logger.info(f"Model checkpoint: {CHECKPOINT}")

# Initialize Hugging Face API
api = HfApi()
logger.info("Hugging Face API initialized")

# Initialize checkpoint manager for failure recovery
checkpoint_manager = CheckpointManager()
logger.info("Checkpoint manager initialized")

# Check for checkpoint to resume from previous run (load BEFORE dataset)
resume_shard_id = checkpoint_manager.get_resume_shard_id()
resume_rows = checkpoint_manager.get_rows_processed()
last_uploaded = checkpoint_manager.get_last_upload()

if resume_shard_id > 0 or resume_rows > 0:
    logger.warning(f"CHECKPOINT DETECTED: Will skip {resume_rows:,} rows and resume from shard {resume_shard_id}")
else:
    logger.info("Starting fresh (no checkpoint found)")

# ============================================================================
# STAGE 4: LOAD DATASET IN STREAMING MODE
# ============================================================================
print("\n" + "=" * 70)
print("STAGE 4: DATASET LOADING")
print("=" * 70)

logger.info(f"Loading dataset from {SOURCE_REPO} in streaming mode...")
stream = load_dataset(SOURCE_REPO, split=SPLIT, streaming=True)

# Skip rows that were already processed in previous runs
if resume_rows > 0:
    logger.info(f"Skipping {resume_rows:,} rows that were already processed...")
    stream = stream.skip(resume_rows)  # ← Skip already processed rows
    logger.info(f"✓ Dataset will start from row {resume_rows + 1:,}")

logger.info("✓ Dataset loaded successfully (streaming mode)")

print(f"✓ Dataset: {SOURCE_REPO}")
print(f"✓ Split: {SPLIT}")
print(f"✓ Mode: Streaming (memory efficient)")
if resume_rows > 0:
    print(f"✓ Starting from row: {resume_rows + 1:,} (skipping {resume_rows:,} processed rows)")
print("=" * 70 + "\n")

# ============================================================================
# STAGE 5: LOAD MODEL AND TOKENIZER
# ============================================================================
print("=" * 70)
print("STAGE 5: MODEL AND TOKENIZER LOADING")
print("=" * 70)

logger.info(f"Loading tokenizer from {CHECKPOINT}...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, padding_side="left")
logger.info("✓ Tokenizer loaded")

logger.info(f"Loading model from {CHECKPOINT}...")
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).to(device)
logger.info(f"✓ Model loaded and moved to device: {device}")

# Log model information
model_device = next(model.parameters()).device
print(f"✓ Tokenizer loaded successfully")
print(f"✓ Model loaded successfully")
print(f"✓ Model device: {model_device}")
print("=" * 70 + "\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_writer(filename: str, schema: pa.Schema) -> ipc.RecordBatchFileWriter:
    """
    Create a PyArrow RecordBatchFileWriter for writing Arrow files.
    
    Args:
        filename (str): Path to output Arrow file
        schema (pa.Schema): Arrow schema for the file
    
    Returns:
        ipc.RecordBatchFileWriter: Writer object for batch writing
    """
    logger.debug(f"Creating writer for file: {filename}")
    return ipc.RecordBatchFileWriter(filename, schema)


def serialize_row_for_arrow(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert row to Arrow-compatible format by serializing non-primitive types.
    
    Handles PIL Image objects by converting them to PNG bytes. This allows
    the row to be stored in Arrow format without type conversion errors.
    
    Args:
        row (Dict[str, Any]): Row potentially containing PIL images and other data
    
    Returns:
        Dict[str, Any]: Row with all values converted to Arrow-compatible types
    """
    serialized_row = {}
    
    for key, value in row.items():
        # Convert PIL Image objects to PNG bytes
        if isinstance(value, Image.Image):
            try:
                img_bytes = BytesIO()
                value.save(img_bytes, format='PNG')
                serialized_row[key] = img_bytes.getvalue()
                logger.debug(f"Serialized image column '{key}' to {len(img_bytes.getvalue())} bytes")
            except Exception as e:
                logger.error(f"Failed to serialize image in column '{key}': {str(e)}")
                raise
        # Keep text, numeric, and other primitive types as-is
        else:
            serialized_row[key] = value
    
    return serialized_row


def batch_augment_rows(rows: list) -> list:
    """
    Augment a batch of dataset rows by rewriting answers as complete sentences.
    
    Uses SmolLM2 LLM to generate expanded versions of answers that provide
    more context and detail while preserving original answers. Processes
    multiple rows in a single batch for efficiency.
    
    Args:
        rows (list): List of dataset rows, each containing:
            - 'question_string': The question asked about the chart
            - 'answer': The original concise answer
    
    Returns:
        list: Rows with added 'augmented_answer' fields
    """
    if not rows:
        return []
    
    system_prompt = f"""
You generate labels for a VQA dataset.

STRICT RULES (the model must obey them):
1. Output EXACTLY ONE SENTENCE in the first line.
2. That sentence MUST explicitly contain the exact answer token: <answer>.
3. The sentence must directly restate the question in a natural, neutral way.
4. The sentence must NOT add any extra information, details, comparisons, or adjectives.
5. The sentence must NOT start with words like "indeed", "clearly", "actually", or "in fact".
6. The sentence must NOT include reasoning or justification.
7. After the sentence, output a new line containing EXACTLY: #### <answer>
8. Output NOTHING else.
"""

    # Prepare batch of chat messages
    messages_list = [
        [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Question: {row['question_string']}\nAnswer: {row['answer']}\n"
            }
        ]
        for row in rows
    ]
    
    # Apply chat template to all messages in batch
    input_texts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True
        )
        for msgs in messages_list
    ]
    
    # Tokenize all inputs at once with padding
    batch_inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate augmented answers for entire batch
    batch_outputs = model.generate(
        input_ids=batch_inputs["input_ids"],
        attention_mask=batch_inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    # Decode all outputs and extract augmented answers
    augmented_rows = []
    for idx, row in enumerate(rows):
        full_response = tokenizer.decode(batch_outputs[idx], skip_special_tokens=False)
        row["augmented_answer"] = full_response
        augmented_rows.append(row)
    
    return augmented_rows

# ============================================================================
# STAGE 6: DATA AUGMENTATION AND STREAMING UPLOAD PIPELINE
# ============================================================================
print("=" * 70)
print("STAGE 6: DATA AUGMENTATION AND STREAMING UPLOAD PIPELINE")
print("=" * 70 + "\n")

# Initialize shard management variables
writer: Optional[ipc.RecordBatchFileWriter] = None
schema: Optional[pa.Schema] = None

if resume_shard_id > 0 or resume_rows > 0:
    logger.warning(f"RESUMING FROM CHECKPOINT: shard_id={resume_shard_id}, rows={resume_rows}, last_upload={last_uploaded}")
    print(f"\n⚠️  RESUMING FROM CHECKPOINT")
    print(f"   Shard ID: {resume_shard_id}")
    print(f"   Rows processed: {resume_rows:,}")
    print(f"   Last upload: {last_uploaded}")
    print()
else:
    logger.info("Starting fresh (no checkpoint found)")
    print(f"\n✓ Starting fresh augmentation pipeline\n")

shard_id = resume_shard_id
rows_in_shard = 0
total_rows_processed = resume_rows

logger.info(f"Starting batch augmentation pipeline (batch_size={BATCH_SIZE}, rows per shard: {ROWS_PER_SHARD})...")

try:
    # Accumulate rows for batch processing
    batch_buffer = []
    write_buffer = []  # Accumulate augmented rows before writing to Arrow
    sample_count = 0
    
    for row in tqdm(stream, desc="Processing samples", unit=" samples"):
        # Check if sample limit has been reached
        if MAX_SAMPLES is not None and sample_count >= MAX_SAMPLES:
            logger.info(f"Sample limit reached ({MAX_SAMPLES:,} samples). Stopping processing.")
            print(f"\n✓ Sample limit reached: {MAX_SAMPLES:,} samples processed")
            break
        
        sample_count += 1
        batch_buffer.append(row)
        
        # Process batch when it reaches desired size
        if len(batch_buffer) >= BATCH_SIZE:
            try:
                logger.debug(f"Augmenting batch of {len(batch_buffer)} rows...")
                # Augment entire batch at once using LLM
                augmented_batch = batch_augment_rows(batch_buffer)
                logger.debug(f"Batch augmentation complete: {len(augmented_batch)} rows")
                
                # Serialize rows for Arrow compatibility
                for aug_row in augmented_batch:
                    aug_row = serialize_row_for_arrow(aug_row)
                    write_buffer.append(aug_row)
                
                batch_buffer = []  # Clear batch buffer
                
                # Write accumulated rows to Arrow when write buffer is full
                if len(write_buffer) >= WRITE_BATCH_SIZE:
                    # Convert all rows to PyArrow record batch at once
                    if write_buffer:
                        # Get all keys from first row
                        all_keys = list(write_buffer[0].keys())
                        # Create arrays for each column
                        data = {key: [row[key] for row in write_buffer] for key in all_keys}
                        write_batch = pa.record_batch(
                            [pa.array(data[col]) for col in all_keys],
                            names=all_keys
                        )
                        
                        # Initialize writer on first write (needed to determine schema)
                        if writer is None:
                            schema = write_batch.schema
                            filename = f"{shard_id:06d}.arrow"
                            writer = create_writer(filename, schema)
                            logger.info(f"Created shard #{shard_id:06d}: {filename} (schema: {len(schema)} fields)")
                        
                        # Write batch to current shard file
                        writer.write_batch(write_batch)
                        rows_in_shard += len(write_buffer)
                        total_rows_processed += len(write_buffer)
                        logger.debug(f"Wrote {len(write_buffer)} rows to shard #{shard_id:06d} (total: {rows_in_shard})")
                        write_buffer = []  # Clear write buffer
                        
                        # Check if shard has reached capacity and needs to be uploaded
                        if rows_in_shard >= ROWS_PER_SHARD:
                            logger.info(
                                f"Shard #{shard_id:06d} full ({rows_in_shard} rows). "
                                f"Closing and uploading to {TARGET_REPO}..."
                            )
                            
                            # Close current shard file
                            writer.close()
                            
                            # Determine filename
                            filename = f"{shard_id:06d}.arrow"
                            
                            # Upload shard to Hugging Face Hub with retry logic
                            max_retries = 3
                            retry_count = 0
                            upload_success = False
                            
                            while retry_count < max_retries and not upload_success:
                                try:
                                    logger.info(f"Uploading shard #{shard_id:06d} to {TARGET_REPO}/{SPLIT}/... (attempt {retry_count + 1}/{max_retries})")
                                    api.upload_file(
                                        repo_id=TARGET_REPO,
                                        repo_type="dataset",
                                        path_or_fileobj=filename,
                                        path_in_repo=f"{SPLIT}/{filename}",
                                    )
                                    logger.info(f"✓ Successfully uploaded {filename}")
                                    upload_success = True
                                    
                                    # Save checkpoint after successful upload
                                    checkpoint_manager.save(shard_id + 1, total_rows_processed, shard_id)
                                    logger.debug(f"Checkpoint saved after uploading shard {shard_id}")
                                    
                                except Exception as upload_error:
                                    retry_count += 1
                                    if retry_count < max_retries:
                                        wait_time = 2 ** retry_count  # Exponential backoff
                                        logger.warning(f"Upload failed (attempt {retry_count}), retrying in {wait_time}s: {str(upload_error)}")
                                        time.sleep(wait_time)
                                    else:
                                        logger.error(f"Upload failed after {max_retries} retries. Checkpoint saved for recovery.")
                                        checkpoint_manager.save(shard_id, total_rows_processed, last_uploaded)
                                        raise
                            
                            # Clean up local file to free disk space only after successful upload
                            if upload_success:
                                os.remove(filename)
                                logger.debug(f"Deleted local file: {filename}")
                                last_uploaded = shard_id
                            
                            # Reset for next shard
                            shard_id += 1
                            rows_in_shard = 0
                            writer = None
            
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}", exc_info=True)
                batch_buffer = []  # Clear buffer on error
                write_buffer = []
                continue
    
    # ========================================================================
    # HANDLE REMAINING BATCH AND WRITE BUFFERS
    # ========================================================================
    # Process any remaining rows in batch buffer
    if batch_buffer:
        logger.info(f"Processing remaining batch with {len(batch_buffer)} rows...")
        try:
            augmented_batch = batch_augment_rows(batch_buffer)
            for aug_row in augmented_batch:
                aug_row = serialize_row_for_arrow(aug_row)
                write_buffer.append(aug_row)
        except Exception as e:
            logger.error(f"Error processing final batch: {str(e)}", exc_info=True)
    
    # Write any remaining rows in write buffer
    if write_buffer:
        logger.info(f"Writing final buffer with {len(write_buffer)} rows...")
        try:
            all_keys = list(write_buffer[0].keys())
            data = {key: [row[key] for row in write_buffer] for key in all_keys}
            final_batch = pa.record_batch(
                [pa.array(data[col]) for col in all_keys],
                names=all_keys
            )
            
            if writer is None:
                schema = final_batch.schema
                filename = f"{shard_id:06d}.arrow"
                writer = create_writer(filename, schema)
                logger.info(f"Created shard #{shard_id:06d}: {filename} (schema: {len(schema)} fields)")
            
            writer.write_batch(final_batch)
            rows_in_shard += len(write_buffer)
            total_rows_processed += len(write_buffer)
            logger.debug(f"Wrote final {len(write_buffer)} rows to shard #{shard_id:06d}")
        except Exception as e:
            logger.error(f"Error writing final buffer: {str(e)}", exc_info=True)
    
    # ========================================================================
    # STAGE 7: HANDLE FINAL SHARD (REMAINING ROWS)
    # ========================================================================
    if writer is not None:
        logger.info(
            f"Processing final shard #{shard_id:06d} with {rows_in_shard} rows..."
        )
        
        # Close current shard file
        writer.close()
        filename = f"{shard_id:06d}.arrow"
        
        # Upload final shard with retry logic
        max_retries = 3
        retry_count = 0
        upload_success = False
        
        while retry_count < max_retries and not upload_success:
            try:
                logger.info(f"Uploading final shard #{shard_id:06d} to {TARGET_REPO}/{SPLIT}/... (attempt {retry_count + 1}/{max_retries})")
                api.upload_file(
                    repo_id=TARGET_REPO,
                    repo_type="dataset",
                    path_or_fileobj=filename,
                    path_in_repo=f"{SPLIT}/{filename}",
                )
                logger.info(f"✓ Successfully uploaded final shard: {filename}")
                upload_success = True
                
            except Exception as upload_error:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    logger.warning(f"Final upload failed (attempt {retry_count}), retrying in {wait_time}s: {str(upload_error)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Final upload failed after {max_retries} retries. Checkpoint saved for recovery.")
                    checkpoint_manager.save(shard_id, total_rows_processed, last_uploaded)
                    raise
        
        # Clean up local file only after successful upload
        if upload_success:
            os.remove(filename)
            logger.debug(f"Deleted local file: {filename}")
        
        total_shards = shard_id + 1
    else:
        total_shards = shard_id
    
    # ========================================================================
    # STAGE 8: FINAL SUMMARY AND COMPLETION
    # ========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"✓ Total rows processed     : {total_rows_processed:,}")
    if MAX_SAMPLES is not None:
        print(f"✓ Sample limit             : {MAX_SAMPLES:,}")
    print(f"✓ Total shards created     : {total_shards}")
    print(f"✓ Target repository        : {TARGET_REPO}/{SPLIT}/")
    print(f"✓ Rows per shard           : {ROWS_PER_SHARD}")
    print("=" * 70)
    
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Total rows processed: {total_rows_processed:,}")
    if MAX_SAMPLES is not None:
        logger.info(f"Sample limit: {MAX_SAMPLES:,}")
    logger.info(f"Total shards created: {total_shards}")
    logger.info(f"Target repository: {TARGET_REPO}/{SPLIT}/")
    logger.info("=" * 70)
    
    # Clear checkpoint after successful completion
    checkpoint_manager.clear()

except KeyboardInterrupt:
    logger.warning("Pipeline interrupted by user (Ctrl+C)")
    if writer is not None:
        writer.close()
        logger.info("Current shard file closed")
    # Save checkpoint for resume
    checkpoint_manager.save(shard_id, total_rows_processed, last_uploaded)
    print("\n⚠️  Pipeline interrupted by user")
    print(f"   Checkpoint saved. Resume with: python {os.path.basename(__file__)}")

except Exception as e:
    logger.critical(f"Pipeline failed with error: {str(e)}", exc_info=True)
    if writer is not None:
        writer.close()
        logger.info("Current shard file closed due to error")
    # Save checkpoint for resume
    checkpoint_manager.save(shard_id, total_rows_processed, last_uploaded)
    print(f"\n❌ Pipeline failed: {str(e)}")
    print(f"   Checkpoint saved. Resume with: python {os.path.basename(__file__)}")
    raise