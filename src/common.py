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
        r'AN\s+ACT\s+TO\s+ESTABLISH\s+A\s+PROVIDENT\s+FUND',text,re.IGNORECASE)

    if not act_heading:raise ValueError('Could not find the EPF Act heading.')

    text = text[act_heading.start():]

    section_one = re.search(r'(?<![\w.])1\.\s+This\s+Act\s+may\s+be\s+cited',
        text,
        re.IGNORECASE)

    if not section_one:
        raise ValueError(
            'Could not find the real Section 1.')

    return text[section_one.start():]

def find_section_matches(text):
    # Find legal section numbers in the EPF Act
    pattern = re.compile(r'(?<![\w.])(\d{1,2}[A-Z]?)\.[ \t]+(?=\S)')

    matches = []
    seen = set()

    for match in pattern.finditer(text):
        number = match.group(1).upper()

        if number not in seen:
            seen.add(number)
            matches.append(match)

    return matches

def extract_sections(text, matches):
    # extract the full text for each detected legal section
    # next section marker is used as the end of the current section
    sections = {}

    for index, match in enumerate(matches):
        number = match.group(1).upper()
        start = match.start()

        end = (matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text))

        sections[f'Section{number}'] = {'section': f'Section{number}','text': text[start:end].strip()}

    return sections

def load_focused_sections():
    # load the Act and break it into separate sections
    # then keep only the sections we use in this focused approach
    text = get_act_body_text()
    matches = find_section_matches(text)
    sections = extract_sections(text, matches)

    focused = []

    for section_name in FOCUSED_SECTIONS:
        if section_name not in sections:
            raise ValueError(f'{section_name} was not found.')

        focused.append(sections[section_name])

    return focused