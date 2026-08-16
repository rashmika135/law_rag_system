import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


#PROJECT PATHS

# get the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# path to the EPF Act PDF
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "epf_act.pdf"

# folder where chromaDB will store the vector database
CHROMA_PATH = PROJECT_ROOT / "vectorstores" / "basic_rag"

#SETTINGS
# basic fixed-size chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# number of chunks retrieved for each question
TOP_K = 5

# Embedding 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# groq 
GROQ_MODEL = "llama-3.1-8b-instant"

# chroma collection name
COLLECTION_NAME = "epf_basic_rag"
