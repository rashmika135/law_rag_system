import os
import re
from pathlib import Path

import chromadb
import pymupdf
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT_DIR / "data" / "raw" / "epf_act.pdf"

FOCUSED_SECTIONS = [
    "Section10",
    "Section11",
    "Section12",
    "Section13",
    "Section14",
    "Section15",
    "Section16",
    "Section17",
    "Section18",
    "Section19",
    "Section20"
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

BASIC_CHUNK_SIZE = 700
BASIC_CHUNK_OVERLAP = 100