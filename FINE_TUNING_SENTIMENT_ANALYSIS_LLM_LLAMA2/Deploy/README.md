# Sentiment Analysis Web-App (Docker)

Gradio app serving the fine-tuned Llama 2 sentiment model (merged, bf16).

## Requirements
- NVIDIA GPU with ~6 GB free memory (model is loaded in 4-bit by default).
  Set `-e LOAD_IN_4BIT=false` to load in bf16 instead (needs ~14 GB).
- Docker + NVIDIA Container Toolkit (`docker run --gpus all ...` must work)

## 1. Download the model from Google Drive
Copy these two folders from
`MyDrive/AI_Learning/1_Generative_AI/Projects/Cap07/saida/saida/model_tuned/final/`
into a `models/` folder next to this README:

```
sentiment-app/
├── app.py
├── Dockerfile
├── requirements.txt
└── models/
    ├── model_tuned_final_bf16/   (~13 GB)
    └── tokenizer_final_bf16/
```

The model is mounted at run time, not copied into the image, so the image stays small.

## 2. Build
```bash
docker build -t sentiment-app .
```

## 3. Run
Linux / macOS:
```bash
docker run --gpus all -p 7860:7860 \
  -v "$(pwd)/models/model_tuned_final_bf16:/models/model:ro" \
  -v "$(pwd)/models/tokenizer_final_bf16:/models/tokenizer:ro" \
  sentiment-app
```

Windows (PowerShell):
```powershell
docker run --gpus all -p 7860:7860 `
  -v "${PWD}/models/model_tuned_final_bf16:/models/model:ro" `
  -v "${PWD}/models/tokenizer_final_bf16:/models/tokenizer:ro" `
  sentiment-app
```

Open http://localhost:7860. Loading the model takes ~1 minute on start-up.

## Without a GPU
Drop `--gpus all`. The app falls back to CPU in float32: it needs ~28 GB of RAM
and each prediction takes several seconds or more.
