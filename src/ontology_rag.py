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