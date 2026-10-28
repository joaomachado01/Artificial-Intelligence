##############################################
############### API MODULE ###################
##############################################

# Import packages
import os
import investment_env_var
from fastapi import FastAPI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings('ignore')
from huggingface_hub.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# Defining Item classe from BaseModel
class Item(BaseModel):
    query: str

# Defining the embedding model
embedding_model_name = "sentence-transformers/msmarco-bert-base-dot-v5"

# Defining embedding model args
embedding_model_kwargs = {'device': 'cpu'}

# Defining encode args
encode_kwargs = {'normalize_embeddings': True}

# Creates a instance of HuggingFaceEmbeddings
hf = HuggingFaceEmbeddings(
    model_name = embedding_model_name,
    model_kwargs = embedding_model_kwargs,
    encode_kwargs = encode_kwargs
)

# Defining the var use_nvidia_api as False
use_nvidia_api = False

# Checks if Nvidia API key is available
if investment_env_var.nvidia_key != "":

    # Imports OpenAI class
    from openai import OpenAI

    # Creates OpenAI instance
    client_ai = OpenAI(base_url = "https://integrate.api.nvidia.com/v1", api_key = investment_env_var.nvidia_key)

    # Defining the var use_nvidia_api as True
    use_nvidia_api = True

else:
    # Message to indicate no use for LLM
    print("1 - Not possible to use LLM!")

### Infra ###
# Create a instance to connect to Vector DB
client_db = QdrantClient(os.getenv("QDRANT_URL", "http://localhost:6333"))

# Collection name
collection_name = "InvestVectorDB"

# Create a instance to send request to Vector DB
qdrant = QdrantVectorStore(client_db, collection_name, hf, distance = Distance.DOT)


### API ###
# Create FastAPI instance
app = FastAPI()

# Defining root for GET method
@app.get("/")
def root():
    return {"message": "Investment project"}

@app.post("/invest_api")
def invest_api(item: Item):

    # Gets item query
    query = item.query

    # Executes similarity search in Vector DB
    search_result = qdrant.similarity_search(query = query, k = 10)

    # Initiates list of results, context and mapping
    list_res = []
    context = ""
    mappings = {}

    # Loop to create entries for the empty vars above
    for i, res in enumerate(search_result):
        context += f"[{i}] \n {res.page_content}\n\n"
        mappings[i] = res.metadata.get("path")
        list_res.append({"id": i, "path": res.metadata.get("path"), "content": res.page_content})

    # Defining system message
    rolemsg = {"role": "system",
               "content": "Answer the user's question using only the documents provided in the context. "
                          "Each document starts with its ID in square brackets, e.g. [0]. "
                          "After every sentence that uses information from a document, cite its ID in square brackets, e.g. [0] or [2]. "
                          "Every answer must contain at least one citation. "
                          "If the answer isn't in the documents, say you don't know."}

    # Defining messages
    messages = [rolemsg, {"role": "user", "content": f"Documents: \n{context}\n\nQuestion: {query}"}]

    # Checks is Nvidia API Key is being used
    if use_nvidia_api:

        # Creates LLM instance using Nvidia API
        response = client_ai.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            messages=messages,
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stream=False
        )

        response_content = response.choices[0].message.content

    else:

        # Prints a message saying that LLM cannot be use
        print("2 - Not possible to use LLM!")

        # Gets response_content when issue happens calling the LLM
        response_content = "LLM not available: NVIDIA API key not configured."

    return {"context": list_res, "answer": response_content}