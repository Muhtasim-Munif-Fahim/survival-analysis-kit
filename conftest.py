"""Make the src tree importable when running pytest from a checkout."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))