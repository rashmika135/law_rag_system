import time

from src.common import (
    load_focused_sections,
    create_section_chunks,
    load_embedding_model,
    create_vector_collection,
    query_vector_collection,
    get_groq_client,
    safe_generate_answer)


TOP_K = 3


def initialize_structure_rag():
    print("Initializing Structure RAG...")

    sections = load_focused_sections()

    chunks = create_section_chunks(
        sections)

    embedding_model = load_embedding_model()

    collection = create_vector_collection( chunks,
        embedding_model,
        "epf_structure_rag")

    rag = { "embedding_model": embedding_model,
        "collection": collection,
        "groq_client": get_groq_client()}

    return rag


def run_structure_query(question, rag):
    start_time = time.perf_counter()

    chunks = query_vector_collection( question,
        rag["collection"],
        rag["embedding_model"],
        top_k=TOP_K)

    answer, error = safe_generate_answer(question,
        chunks,
        rag["groq_client"] )

    latency = time.perf_counter() - start_time

    return { "answer": answer,
        "retrieved_chunks": chunks,
        "latency": latency,
        "generation_error": error }


if __name__ == "__main__":
    rag = initialize_structure_rag()

    while True:
        question = input( "\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        result = run_structure_query( question, rag)

        print("\nRetrieved:")

        for chunk in result["retrieved_chunks"]:
            print(chunk["sections"] )

        print("\nAnswer:")
        print( result["answer"] )