from pathlib import Path
import fitz


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