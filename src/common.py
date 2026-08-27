from pathlib import Path


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT_DIR / 'data' / 'raw' / 'epf_act.pdf'


# sections of the epf act used in the focused approcah
FOCUSED_SECTIONS = ['Section10', 'Section11', 'Section12','Section13', 'Section14', 
                    'Section15','Section16', 'Section17', 'Section18','Section19', 'Section20'
]