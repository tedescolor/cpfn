from pathlib import Path
import sys

# Ensure project `src/` is on sys.path so tests can import the package
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
