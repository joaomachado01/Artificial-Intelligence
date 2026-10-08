"""Sentiment Analysis Web-App - Llama 2 fine-tuned with QLoRA (merged bf16 model)."""
import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Paths inside the container (the model folders are mounted as volumes)
MODEL_DIR = os.getenv("MODEL_DIR", "/models/model")
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "/models/tokenizer")
PORT = int(os.getenv("PORT", "7860"))
# 4-bit loading (~5 GB of GPU memory) for GPUs with less than ~14 GB
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"

# GPU: 4-bit (default, ~5 GB) or bf16 (~14 GB). No GPU: CPU in float32 (slow, ~28 GB RAM)
load_kwargs = {}
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    load_kwargs["device_map"] = "auto"
    if LOAD_IN_4BIT:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
else:
    print("WARNING: no GPU detected, running on CPU (slow).")
    load_kwargs["torch_dtype"] = torch.float32

print(f"Loading model from {MODEL_DIR} (4-bit={LOAD_IN_4BIT and torch.cuda.is_available()}) ...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **load_kwargs)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
print("Model loaded.")


def predict_sentiment(text, max_new_tokens=5):
    # Same format used in training (the tokenizer adds <s> automatically)
    prompt = f"[INST] {text} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()

    if "positive" in generated.lower():
        return "Positive"
    if "negative" in generated.lower():
        return "Negative"
    return f"Unknown ({generated})"


def return_sentiment(entry):
    if not entry or not entry.strip():
        return "Please enter a sentence."
    label = predict_sentiment(entry)
    if label == "Positive":
        return "Positive 😊"
    if label == "Negative":
        return "Negative 😞"
    return label


webapp = gr.Interface(
    fn=return_sentiment,
    inputs=gr.Textbox(label="Tell me the sentence to classify the sentiment",
                      lines=1, info="Put a sentence however you like"),
    outputs=gr.Textbox(label="Result (model prediction)", lines=1),
    title="Sentiment Analysis Web-App",
    description="Give the entries for the model and click the button to see its sentiment",
    examples=["I feel like trying to cope with AI is exhausting", "I don't feel well today"],
    flagging_mode="never",
)

if __name__ == "__main__":
    # 0.0.0.0 so the app is reachable from outside the container
    webapp.launch(server_name="0.0.0.0", server_port=PORT)
