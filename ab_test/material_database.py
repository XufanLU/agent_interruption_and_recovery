from __future__ import annotations

from typing import Any

from fixture_pairs import FIXTURES_DIR, get_fixture_pair, load_fixture_pairs


def search_materials(chemsys: str) -> str | None:
    query = str(chemsys or "").strip().lower()
    normalized = query.replace("-", "")
    for pair in load_fixture_pairs():
        material_id = str(pair.get("material_id", ""))
        formula = str(pair.get("formula", ""))
        if query in {material_id.lower(), formula.lower()} or normalized == formula.lower():
            return material_id
    return None


def get_material_by_id(material_id: str) -> dict[str, Any] | None:
    pair = get_fixture_pair(material_id)
    cif_path = pair.get("cif_path") if pair else f"{material_id}.cif"
    path = FIXTURES_DIR / str(cif_path)
    if not path.exists():
        return None
    return {"material_id": material_id, "file_path": str(path)}


def search_material_candidates(chemsys: str) -> list[dict[str, Any]]:
    material_id = search_materials(chemsys)
    if not material_id:
        return []
    pair = get_fixture_pair(material_id) or {}
    return [{"material_id": material_id, "formula": pair.get("formula", material_id)}]
