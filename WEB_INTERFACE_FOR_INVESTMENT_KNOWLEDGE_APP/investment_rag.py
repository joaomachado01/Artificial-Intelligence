##############################################
############### RAG MODULE ###################
##############################################

# Import packages for the RAG module
import os
import sys
import docx 
import PyPDF2
from pptx import Presentation
from os import listdir
from os.path import isfile, join, isdir
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
import warnings
warnings.filterwarnings('ignore')
from huggingface_hub.utils import logging as hf_logging
hf_logging.set_verbosity_error()


# Function to create list with files path
def create_files_list(dir):
    # Empty list
    files_list = []

    # Iterate on dir path
    for file in listdir(dir):

        # If is a file, adds to list
        if isfile(join(dir, file)):
            files_list.append(join(dir, file))

        # If directory, iterate within to bring files
        elif isdir(join(dir, file)):
            files_list += create_files_list(join(dir, file))

    # Returns list
    return files_list

def extract_word_text(file):
    # Open word
    doc = docx.Document(file)

    # Extract docs paragraph list
    word_text_list = [para.text for para in doc.paragraphs]

    # Return as 1 big text string
    return '\n'.join(word_text_list)

def extract_ppt_text(file):
    # Open ppt
    ppt = Presentation(file)

    # Empty list for ppt text
    ppt_text_list = []

    # Iterate slides
    for slide in ppt.slides:

        # Iterate each shape
        for shape in slide.shapes:

            # If is text attribute then appends attribute text
            if hasattr(shape, "text"):
                ppt_text_list.append(shape.text)

    # Return as 1 big text string
    return '\n'.join(ppt_text_list)

def main_indexing(dirpath):

    # Defining the model name for the embeddings generator
    model_name = "sentence-transformers/msmarco-bert-base-dot-v5"

    # Defining model configs
    model_kwargs = {'device': 'cpu'}

    # Defining encode configs
    encode_kwargs = {'normalize_embeddings': True}

    # Initiate embeddings instance
    hf = HuggingFaceEmbeddings(model_name = model_name,
                               model_kwargs = model_kwargs,
                               encode_kwargs = encode_kwargs)

    # Initiate Qdrant client
    client = QdrantClient(os.getenv("QDRANT_URL", "http://localhost:6333"))

    # Defining collection name
    collection_name = "InvestVectorDB"

    # If collection extists, drop
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    # Create collection
    client.create_collection(collection_name,
                             vectors_config = VectorParams(size = 768, distance = Distance.DOT))

    # Initiate Qdrant instance
    qdrant = QdrantVectorStore(client, collection_name, hf, distance = Distance.DOT)

    # Message to inform indexing is starting
    print("\nIndexing document..\n")

    # Get all files from current dir
    files_list = create_files_list(dirpath)

    # Iterate on each doc
    for file in files_list:

        try:
            # Save empty string to hold text
            file_content = ""
            
            # If file is PDF
            if file.endswith(".pdf"):
                
                print("Indexing: " + file)
                
                reader = PyPDF2.PdfReader(file)
                
                for page in reader.pages:
                    file_content += " " + (page.extract_text() or "")
            
            # If file is TXT
            elif file.endswith(".txt"):
                
                print("Indexing: " + file)
                
                with open(file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            
            # If file is WORD
            elif file.endswith(".docx"):
                
                print("Indexing: " + file)
                
                file_content = extract_word_text(file)
            
            # If file is PPT
            elif file.endswith(".pptx"):
                
                print("Indexing: " + file)
                
                file_content = extract_ppt_text(file)
            
            else:
                
                # If file format is not supported, continue on the next one
                continue

            # Initiate text splitter
            text_splitter = TokenTextSplitter(chunk_size = 500, chunk_overlap = 50)

            # Slit text into chunks
            chunks = text_splitter.split_text(file_content)

            # Creating metadata to be after pair with the chunks
            metadata = [{"path":file} for _ in chunks]

            # Adding chunks into Qdrant instance
            qdrant.add_texts(chunks, metadatas = metadata)

        except Exception as e:

            # If issue indexing file
            print(f"The process failed for the file {file}: {e}")

    # Print execution end
    print("\nIndexing finished!\n")

if __name__ == "__main__":

    # Get args from CLI
    args = sys.argv

    # Execute
    if len(args) > 1:
        main_indexing(args[1])

    else:
        # For error
        print("ERROR: You need to add a document folder path to index!")