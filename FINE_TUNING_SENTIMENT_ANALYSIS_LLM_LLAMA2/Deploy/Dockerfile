# Base image already includes Python, CUDA 12.8 and torch 2.7.1
# (CUDA 12.8 is required for RTX 50-series / Blackwell GPUs)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    MODEL_DIR=/models/model \
    TOKENIZER_DIR=/models/tokenizer \
    PORT=7860 \
    LOAD_IN_4BIT=true

# C compiler: triton (used by bitsandbytes) compiles a small CUDA helper at start-up
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
