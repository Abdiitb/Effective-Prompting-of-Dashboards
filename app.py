import gradio as gr
import torch
from transformers import BitsAndBytesConfig, Idefics3ForConditionalGeneration, AutoProcessor
from dataclasses import dataclass
from typing import Any, Dict, List

# Load your fine-tuned model
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)

# Enables 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Double quantization for better compression
    bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in bfloat16 for stability
)

# Load pre-trained VLM model from local directory
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",  # Automatically map model layers to available devices
    dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    quantization_config=bnb_config,  # Uncomment to enable quantization
    _attn_implementation="sdpa",  # Scaled Dot Product Attention (efficient)
)

adapter_path = "Abd223653/SmolVLM_Finetune_PlotQA"
model.load_adapter(adapter_path)

system_message = """
You are a Vision-Language Model specialized in interpreting chart and plot images.
Analyze the chart carefully and answer the given question concisely (usually a single word, number, or short phrase).
Use both visual information (values, colors, labels) and simple reasoning (e.g., finding averages, differences, trends) based on the chart data.
Do not rely on any external or prior knowledge — all answers must come from interpreting the chart itself.
"""

# Define the chat function
def vlm_chat(history, image, message):
    if image is None:
        return history + [("⚠️ Please upload an image first!", None)], history
    if not message.strip():
        return history + [("⚠️ Please type a question!", None)], history

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {message}"},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Prepare inputs
    inputs = processor(text=text, images=image, return_tensors="pt").to("cuda")

    # Generate response
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=150)
    response = processor.decode(output[0], skip_special_tokens=True)

    # Update chat history
    history.append((message, response.split("Assistant: ")[-1]))
    return history, history

# Define Gradio UI
with gr.Blocks(theme=gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    text_size=gr.themes.sizes.text_md,
    font="Inter",
)) as demo:
    gr.Markdown(
        """
        <div align="center">
            <h1 style="color:#93c5fd;">🧠 Vision-Language Chat Demo</h1>
            <p style="color:#cbd5e1;">Upload an image and chat with your fine-tuned VLM.</p>
        </div>
        """,
    )

    with gr.Row():
        with gr.Column(scale=0.7):
            chatbot = gr.Chatbot(
                label="Chat History",
                bubble_full_width=False,
                height=500,
                show_label=False
            )
            msg = gr.Textbox(
                placeholder="Ask something about the image...",
                show_label=False,
                lines=2,
            )
            send_btn = gr.Button("Send 💬", variant="primary")

        with gr.Column(scale=0.3):
            image_input = gr.Image(
                type="pil", label="Upload Image", height=300
            )

    # Clear button
    clear_btn = gr.Button("Clear Chat 🧹", variant="secondary")

    # Event bindings
    send_btn.click(vlm_chat, inputs=[chatbot, image_input, msg], outputs=[chatbot, chatbot])
    msg.submit(vlm_chat, inputs=[chatbot, image_input, msg], outputs=[chatbot, chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)

# Launch
demo.launch()