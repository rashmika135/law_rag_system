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