# conftest.py — pytest root configuration
# Adds src/ to sys.path so tests can import from it directly.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
