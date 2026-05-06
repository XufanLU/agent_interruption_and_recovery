from __future__ import annotations

from fixture_pairs import get_fixture_pair, load_fixture_pairs, resolve_fixture_path


DEFAULT_DATASET_ID = "cuo_mp-14549_nims-cu-k"


def get_datasets() -> dict[str, tuple[str, str]]:
    datasets: dict[str, tuple[str, str]] = {}
    for pair in load_fixture_pairs():
        formula = str(pair.get("formula", ""))
        edge = str(pair.get("edge", ""))
        element = str(pair.get("absorbing_element", ""))
        name = f"A/B test {formula} {element} {edge}-edge XAS spectrum".strip()
        datasets[name] = (str(pair["id"]), formula)
    return datasets


def get_data_by_id(dataset_id: str) -> list[str]:
    pair = get_fixture_pair(dataset_id)
    if pair:
        xas_path = resolve_fixture_path(str(pair["xas_path"]))
        if xas_path.exists():
            return [str(xas_path)]
    return []
