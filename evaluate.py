import csv
import sys
from pathlib import Path

from src.common import ( load_focused_sections,
    create_basic_chunks,
    create_section_chunks,
    load_embedding_model,
    create_vector_collection,
    query_vector_collection)

from src.ontology_rag import (TOP_K,
    CANDIDATE_K,
    load_ontology_concepts,
    detect_ontology_concepts,
    get_ontology_sections,
    rerank_with_ontology)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = "evaluation_questions.csv"


def load_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def get_expected_sections(row):
    sections = row["expected_sections"].split("|")
    expected = set()

    for section in sections:
        section = section.strip()

        if section:
            expected.add(section)

    return expected


def get_retrieved_sections(chunks):
    section_sets = []

    for chunk in chunks:
        section_sets.append( set(chunk.get("sections", [])))

    return section_sets


def calculate_metrics(chunks, expected):
    retrieved = get_retrieved_sections(chunks)

    # Hit@1
    hit_1 = 0

    if retrieved:
        if retrieved[0] & expected:
            hit_1 = 1

    # Hit@3
    top_3 = retrieved[:3]
    hit_3 = 0

    for sections in top_3:
        if sections & expected:
            hit_3 = 1
            break

    # MRR
    mrr = 0.0

    for i, sections in enumerate(retrieved):
        if sections & expected:
            rank = i + 1
            mrr = 1 / rank
            break

    # Recall@3
    found_sections = set()

    for sections in top_3:
        found_sections.update(sections)

    if expected:
        correct_found = found_sections & expected
        recall_3 = len(correct_found) / len(expected)
    else:
        recall_3 = 0.0

    return hit_1, hit_3, mrr, recall_3


def sections_text(chunks):
    result = []

    for chunk in chunks:
        sections = ",".join( chunk.get("sections", []))

        result.append(sections)

    return " > ".join(result)


def initialize_systems():
    print("Loading sections and embedding model...")

    sections = load_focused_sections()
    embedding_model = load_embedding_model()

    # Basic RAG chunks
    basic_chunks = create_basic_chunks(sections)
    # Section-aware chunks
    section_chunks = create_section_chunks( sections)

    basic_collection = create_vector_collection( basic_chunks, embedding_model,"epf_eval_basic")

    structure_collection = create_vector_collection(section_chunks, embedding_model,"epf_eval_structure")

    ontology_collection = create_vector_collection( section_chunks, embedding_model, "epf_eval_ontology"
)

    graph, concepts = load_ontology_concepts( embedding_model)
    
    return {"embedding_model": embedding_model,
        "basic_collection": basic_collection,
        "structure_collection": structure_collection,
        "ontology_collection": ontology_collection,
        "graph": graph,
        "concepts": concepts}


def retrieve_basic(question, systems):
    return query_vector_collection(question,
        systems["basic_collection"],
        systems["embedding_model"],
        top_k=TOP_K)


def retrieve_structure(question, systems):
    return query_vector_collection( question,
        systems["structure_collection"],
        systems["embedding_model"],
        top_k=TOP_K)


def retrieve_ontology(question, systems):
    candidates = query_vector_collection( question,
        systems["ontology_collection"],
        systems["embedding_model"],
        top_k=CANDIDATE_K)

    matches = detect_ontology_concepts(question,
        systems["embedding_model"],
        systems["concepts"])

    direct_sections, related_sections = get_ontology_sections( matches, systems["graph"])

    reranked = rerank_with_ontology( candidates,
        direct_sections,
        related_sections)

    return reranked[:TOP_K]


def main():
    # Choose question file
    if len(sys.argv) > 1:
        question_file = sys.argv[1]
    else:
        question_file = DEFAULT_QUESTIONS

    questions_path = ROOT_DIR / question_file

    if not questions_path.exists():
        raise FileNotFoundError(f"Question file not found: {questions_path}" )

    results_path = ROOT_DIR / ( questions_path.stem + "_results.csv")

    questions = load_questions( questions_path)

    systems = initialize_systems()

    totals = {
        "Basic RAG": [0, 0, 0, 0],
        "Structure RAG": [0, 0, 0, 0],
        "Ontology RAG": [0, 0, 0, 0]
    }

    result_rows = []

    print()
    print("Question file:", questions_path.name)
    print("Evaluating", len(questions), "questions...")
    print()

    for index, row in enumerate(questions, start=1):
        question = row["question"]

        expected = get_expected_sections(row)

        basic_chunks = retrieve_basic( question, systems)

        structure_chunks = retrieve_structure( question, systems)

        ontology_chunks = retrieve_ontology( question, systems )

        outputs = {"Basic RAG": basic_chunks,"Structure RAG": structure_chunks,"Ontology RAG": ontology_chunks}

        print( f"{index:02d}.",
            row["type"],
            "Expected:",
            ", ".join(sorted(expected)) )

        for system_name, chunks in outputs.items():
            hit_1, hit_3, mrr, recall_3 = calculate_metrics( chunks, expected )

            totals[system_name][0] += hit_1
            totals[system_name][1] += hit_3
            totals[system_name][2] += mrr
            totals[system_name][3] += recall_3

            retrieved_text = sections_text( chunks )

            print( "   ", system_name, retrieved_text )

            result_rows.append({
                "id": row["id"],
                "type": row["type"],
                "question": question,
                "expected_sections": row["expected_sections"],
                "system": system_name,
                "retrieved": retrieved_text,
                "hit_at_1": hit_1,
                "hit_at_3": hit_3,
                "mrr": round(mrr, 4),
                "recall_at_3": round(recall_3, 4)})

        print()

    question_count = len(questions)

    print("=" * 66)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 66)

    print(
        f'{"System":<16}'
        f'{"Hit@1":>10}'
        f'{"Hit@3":>10}'
        f'{"MRR":>10}'
        f'{"Recall@3":>12}'
    )

    for system_name, values in totals.items():
        hit_1 = values[0] / question_count
        hit_3 = values[1] / question_count
        mrr = values[2] / question_count
        recall_3 = values[3] / question_count

        print( f"{system_name:<16}"
            f"{hit_1:>10.3f}"
            f"{hit_3:>10.3f}"
            f"{mrr:>10.3f}"
            f"{recall_3:>12.3f}")

    fieldnames = [ "id",
        "type",
        "question",
        "expected_sections",
        "system",
        "retrieved",
        "hit_at_1",
        "hit_at_3",
        "mrr",
        "recall_at_3"]

    with open( results_path, "w",newline="",encoding="utf-8" ) as file:

        writer = csv.DictWriter( file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(result_rows)

    print()
    print("Detailed results saved to:",results_path)


if __name__ == "__main__":
    main()