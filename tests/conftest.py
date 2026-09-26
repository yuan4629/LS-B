"""Run the tests from the repository root with the root on sys.path."""
import os
import sys
from pathlib import Path

# torch and MKL-linked numpy can load two OpenMP runtimes on some hosts.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
