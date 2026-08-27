from pathlib import Path
import fitz
import re

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT_DIR / 'data' / 'raw' / 'epf_act.pdf'


# sections of the epf act used in the focused approcah
FOCUSED_SECTIONS = ['Section10', 'Section11', 'Section12','Section13', 'Section14', 
                    'Section15','Section16', 'Section17', 'Section18','Section19', 'Section20'
]

# PDF reading
def read_pdf_text():
    # Load text from the EPF Act PDF
    if not PDF_PATH.exists():
        raise FileNotFoundError( f'PDF not found: {PDF_PATH}')

    pdf = fitz.open(str(PDF_PATH))

    text = '\n'.join(page.get_text('text', sort=True)for page in pdf)

    pdf.close()

    return text

def get_act_body_text():
    # Find the beginning of the actual EPF Act
    text = read_pdf_text()

    act_heading = re.search(
        r'AN\s+ACT\s+TO\s+ESTABLISH\s+A\s+PROVIDENT\s+FUND',
        text,
        re.IGNORECASE
    )

    if not act_heading:
        raise ValueError(
            'Could not find the EPF Act heading.'
        )

    text = text[act_heading.start():]

    section_one = re.search(
        r'(?<![\w.])1\.\s+This\s+Act\s+may\s+be\s+cited',
        text,
        re.IGNORECASE
    )

    if not section_one:
        raise ValueError(
            'Could not find the real Section 1.'
        )

    return text[section_one.start():]