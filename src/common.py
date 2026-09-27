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

def load_focused_sections():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pdf = pymupdf.open(PDF_PATH)

    page_texts = []

    for page in pdf:
        page_texts.append(page.get_text("text", sort=True))

    pdf.close()

    text = "\n".join(page_texts)

    # Find the real beginning of the EPF Act
    act_match = re.search(
        r"AN\s+ACT\s+TO\s+ESTABLISH\s+A\s+PROVIDENT\s+FUND",
        text,
        re.IGNORECASE)

    if not act_match:
        raise ValueError("Could not find the EPF Act heading.")

    text = text[act_match.start():]

    # Find the real Section 1
    section_one = re.search(
        r"(?<![\w.])1\.\s+This\s+Act\s+may\s+be\s+cited",
        text,
        re.IGNORECASE)

    if not section_one:
        raise ValueError("Could not find the real Section 1.")

    text = text[section_one.start():]

    # Find section numbers such as 10., 11., 12.
    section_pattern = re.compile(r"(?<![\w.])(\d{1,2}[A-Z]?)\.[ \t]+(?=\S)")

    matches = []
    seen = set()

    for match in section_pattern.finditer(text):
        number = match.group(1).upper()

        if number not in seen:
            seen.add(number)
            matches.append(match)

    all_sections = {}

    for i in range(len(matches)):
        match = matches[i]

        number = match.group(1).upper()
        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        section_name = f"Section{number}"

        all_sections[section_name] = {
            "section": section_name,
            "text": text[start:end].strip()}

    focused_sections = []

    for section_name in FOCUSED_SECTIONS:
        if section_name not in all_sections:
            raise ValueError(
                f"{section_name} was not found.")

        focused_sections.append(
            all_sections[section_name])

    return focused_sections

def create_basic_chunks(
    sections,
    chunk_size=BASIC_CHUNK_SIZE,
    overlap=BASIC_CHUNK_OVERLAP):
    full_text = ""
    section_positions = []

    for item in sections:
        if full_text:
            full_text += "\n\n"

        start = len(full_text)

        full_text += item["text"]

        end = len(full_text)

        section_positions.append({
            "section": item["section"],
            "start": start,
            "end": end})

    chunks = []

    start = 0

    while start < len(full_text):
        end = min(
            start + chunk_size,
            len(full_text))

        chunk_sections = []

        for item in section_positions:
            if start < item["end"] and end > item["start"]:
                chunk_sections.append( item["section"])

        chunks.append({"id": f"basic_{len(chunks)}",
            "text": full_text[start:end].strip(),
            "sections": chunk_sections})

        if end == len(full_text):
            break

        start = end - overlap

    return chunks

def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)