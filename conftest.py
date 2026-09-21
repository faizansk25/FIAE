import os
import sys
from pathlib import Path

# Allow running the test suite from a source checkout without installation.
#
# The release workflow sets FIAE_TEST_INSTALLED=1 to verify the *installed
# wheel*: prepending src/ here would shadow site-packages and make the
# verification meaningless.
if not os.environ.get("FIAE_TEST_INSTALLED"):
    _SRC = Path(__file__).resolve().parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
