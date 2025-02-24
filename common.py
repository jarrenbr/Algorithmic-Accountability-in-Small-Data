import sys
from pathlib import Path

# DEBUG = sys.gettrace() is not None
DEBUG = False

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)
