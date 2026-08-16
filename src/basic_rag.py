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

# embedding 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# groq 
GROQ_MODEL = "llama-3.1-8b-instant"

# chroma collection name
COLLECTION_NAME = "epf_basic_rag"

#load env variab 
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY was not found"
    )


#read epf act pdf
def extract_pdf_pages(pdf_path):

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        # get text from the current page
        text = page.get_text("text")

        pages.append({
            "page": page_number + 1,
            "text": text})
    document.close()

    return pages

# text chunking
def create_basic_chunks(pages):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    chunk_id = 0

    for page in pages:

        # Split the page text
        page_chunks = text_splitter.split_text(page["text"])

        for text in page_chunks:

            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": text,
                "page": page["page"]})
            chunk_id += 1

    return chunks

#embedding model
def load_embedding_model():

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding model loaded.")

    return model

# creating the vector store
def create_vector_store(chunks, embedding_model):

    print("\nCreating embeddings...")

    chunk_texts = [chunk["text"] for chunk in chunks]

    # Convert document chunks into embeddings
    embeddings = embedding_model.encode_document(
        chunk_texts,
        show_progress_bar=True
    )
    print("Embeddings created.")

    # Create ChromaDB database
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Delete the old collection if it already exists.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    # Metadata stores the PDF page for each chunk
    metadatas = [{"page": chunk["page"]}for chunk in chunks]

    ids = [chunk["id"]for chunk in chunks]

    # Store text + embeddings + metadata
    collection.add(
        ids=ids,
        documents=chunk_texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(f"Stored {collection.count()} chunks in ChromaDB.")
    return collection

#retreiving chunks relevant to the questions
def retrieve_chunks(
    question,
    collection,
    embedding_model):
    # Convert the question into an embedding
    question_embedding = embedding_model.encode_query(
        question
    )
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=TOP_K,
        include=["documents","metadatas","distances"])

    retrieved_chunks = []
    for i in range(len(results["documents"][0])):

        retrieved_chunks.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i]})

    return retrieved_chunks