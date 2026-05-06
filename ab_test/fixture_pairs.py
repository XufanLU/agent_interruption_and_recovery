from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
PAIR_MANIFEST_PATH = FIXTURES_DIR / "xas_cif_pairs.json"


@lru_cache(maxsize=1)
def load_fixture_pairs() -> list[dict[str, Any]]:
    manifest = json.loads(PAIR_MANIFEST_PATH.read_text(encoding="utf-8"))
    return list(manifest.get("pairs", []))


def get_fixture_pair(pair_id: str) -> dict[str, Any] | None:
    target = str(pair_id or "").strip().lower()
    if not target:
        return None
    for pair in load_fixture_pairs():
        identifiers = {
            str(pair.get("id", "")).lower(),
            str(pair.get("material_id", "")).lower(),
            str(pair.get("formula", "")).lower(),
        }
        if target in identifiers:
            return pair
    return None


def resolve_fixture_path(relative_path: str) -> Path:
    return FIXTURES_DIR / relative_path
