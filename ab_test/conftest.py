from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
AB_TEST_DIR = Path(__file__).resolve().parent

for path in (str(AB_TEST_DIR), str(BACKEND_DIR)):
    while path in sys.path:
        sys.path.remove(path)

# Keep this folder first on sys.path so the vendored `function_calling/`
# package and local A/B helper modules win over similarly named backend modules.
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(AB_TEST_DIR))
