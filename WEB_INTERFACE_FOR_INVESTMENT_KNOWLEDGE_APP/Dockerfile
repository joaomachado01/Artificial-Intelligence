# DOCKER
# One image for the whole app: the API, the web app and the indexing job
# all run from it, each with a different command (see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) CPU-only torch first. The default Linux torch from PyPI pulls in
#    several GB of NVIDIA CUDA libraries that this app never uses.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

# 2) The rest of the dependencies (torch is already satisfied, so pip skips it)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Download the embedding model at build time, so the container
#    doesn't need to fetch it from Hugging Face on every start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')"

# 4) Application code and the documents to index
COPY investment_api.py investment_env_var.py investment_rag.py \
     investment_start_api.py investment_web_app.py index_entrypoint.py ./
COPY Documents ./Documents
COPY assets ./assets

EXPOSE 8000 8501

# Default command: start the API (overridden for the web app and indexer)
CMD ["python", "investment_start_api.py"]