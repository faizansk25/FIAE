import sys
from pathlib import Path

# Allow running the test suite from a source checkout without installation.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
