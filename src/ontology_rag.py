import time

import numpy as np
from rdflib import Graph, Namespace, RDFS
from rdflib.namespace import SKOS

from src.common import (
    ROOT_DIR,
    load_focused_sections,
    create_section_chunks,
    load_embedding_model,
    create_vector_collection,
    query_vector_collection,
    get_groq_client,
    safe_generate_answer)


ONTOLOGY_PATH = ROOT_DIR / "ontology" / "epf_ontology.rdf"

TOP_K = 3
CANDIDATE_K = 11
ONTOLOGY_TOP_K = 3

DIRECT_ONTOLOGY_BOOST = 0.20
RELATED_ONTOLOGY_BOOST = 0.10
CONCEPT_THRESHOLD = 0.30

EPF = Namespace("http://example.org/epf#")

def name(uri):
    return str(uri).split("#")[-1]


def load_ontology_concepts(embedding_model):
    if not ONTOLOGY_PATH.exists():
        raise FileNotFoundError(
            f"Ontology not found: {ONTOLOGY_PATH}")

    graph = Graph()
    graph.parse(ONTOLOGY_PATH)

    concepts = []

    for concept, _, section in graph.triples(
        (None, EPF.definedIn, None)):
        labels = []
        alt_labels = []

        for label in graph.objects(
            concept,
            RDFS.label):
            labels.append(
                str(label) )

        for label in graph.objects(
            concept,
            SKOS.altLabel):
            alt_labels.append(
                str(label) )

        text_parts = [ name(concept)]

        text_parts.extend( labels)

        text_parts.extend( alt_labels)

        concepts.append({"concept": name(concept),
            "section": name(section),
            "text": " ".join(text_parts) })

    if len(concepts) == 0:
        raise ValueError("No ontology concepts using epf:definedIn were found." )

    texts = []

    for concept in concepts:
        texts.append(concept["text"])

    embeddings = embedding_model.encode(texts, normalize_embeddings=True )

    for i in range(len(concepts)):
        concepts[i]["embedding"] = embeddings[i]

    return graph, concepts

def detect_ontology_concepts(question,
    embedding_model,
    concepts,
    top_k=ONTOLOGY_TOP_K):
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True)[0]

    matches = []

    for concept in concepts:
        score = float(
            np.dot( question_embedding, concept["embedding"]  ) )

        if score >= CONCEPT_THRESHOLD:
            matches.append({ "concept": concept["concept"],
                "section": concept["section"],
                "score": score })

    matches.sort(key=lambda item: item["score"],reverse=True )

    return matches[:top_k]

def get_related_concepts(concept_name, graph):
    concept = EPF[concept_name]
    related = set()

    relations = [ EPF.hasConsequence, EPF.relatedTo]

    for relation in relations:
        for target in graph.objects(concept, relation):
            related.add( name(target))

        for source in graph.subjects(relation, concept):
            related.add(name(source) )

    related.discard( concept_name)

    return related

def get_concept_sections( concept_name,graph):
    sections = set()

    concept = EPF[concept_name]

    for section in graph.objects( concept, EPF.definedIn):
        sections.add( name(section) )

    return sections

def get_ontology_sections(matches,graph):
    direct_sections = set()
    related_sections = set()

    for match in matches:
        direct_sections.add( match["section"])

        related_concepts = get_related_concepts( match["concept"],graph )

        for concept_name in related_concepts:
            sections = get_concept_sections(concept_name,graph)

            related_sections.update(sections)

    related_sections = ( related_sections- direct_sections )

    return ( direct_sections,related_sections)

def rerank_with_ontology( chunks, direct_sections, related_sections):
    reranked = []

    for chunk in chunks:
        vector_score = chunk["vector_similarity"]
        bonus = 0.0

        if chunk["sections"]:
            section = chunk["sections"][0]
        else:
            section = None

        if section in direct_sections:
            bonus = DIRECT_ONTOLOGY_BOOST

        elif section in related_sections:
            bonus = RELATED_ONTOLOGY_BOOST

        new_chunk = chunk.copy()

        new_chunk["ontology_bonus"] = bonus

        new_chunk["final_score"] = (vector_score + bonus)

        reranked.append( new_chunk )

    reranked.sort(key=lambda item: item["final_score"],
        reverse=True)

    return reranked

def initialize_ontology_rag():
    print("Initializing Ontology RAG...")

    sections = load_focused_sections()

    chunks = create_section_chunks(sections)

    embedding_model = load_embedding_model()

    collection = create_vector_collection(chunks, embedding_model,"epf_ontology_rag")

    graph, concepts = load_ontology_concepts( embedding_model)

    rag = { "embedding_model": embedding_model,
        "collection": collection,
        "groq_client": get_groq_client(),
        "concepts": concepts,
        "graph": graph}

    return rag

def run_ontology_query(question, rag):
    start_time = time.perf_counter()

    candidates = query_vector_collection( question,
        rag["collection"],
        rag["embedding_model"],
        top_k=CANDIDATE_K)

    ontology_matches = detect_ontology_concepts(question,
        rag["embedding_model"],
        rag["concepts"])

    direct_sections, related_sections = get_ontology_sections(ontology_matches, rag["graph"])

    chunks = rerank_with_ontology(candidates,
        direct_sections,
        related_sections)

    chunks = chunks[:TOP_K]

    answer, error = safe_generate_answer( question,
        chunks,
        rag["groq_client"])

    latency = (time.perf_counter() - start_time)

    return { "answer": answer,
        "retrieved_chunks": chunks,
        "latency": latency,
        "generation_error": error,
        "ontology_matches": ontology_matches,
        "direct_sections": sorted(direct_sections),
        "related_sections": sorted(related_sections)}
    
if __name__ == "__main__":
    rag = initialize_ontology_rag()

    while True:
        question = input( "\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        result = run_ontology_query( question, rag)
        print("\nOntology matches:")
        print( result["ontology_matches"])

        print("\nDirect ontology sections:")
        print( result["direct_sections"])

        print("\nRelated ontology sections:")
        print( result["related_sections"])

        print("\nFinal retrieved:")

        for chunk in result["retrieved_chunks"]:
            print(chunk["sections"],
                "| vector =",
                round( chunk["vector_similarity"],3),
                "| ontology =",
                round( chunk["ontology_bonus"],3 ),
                "| final =",
                round( chunk["final_score"],  3 ))

        print("\nAnswer:")
        print( result["answer"])