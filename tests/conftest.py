"""Pytest configuration.

Adds the project root to sys.path so `worker.handler` is importable
in tests without installing worker as a package (it runs inside Docker).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
