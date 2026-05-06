from __future__ import annotations

from ab_test.fixture_pairs import get_fixture_pair, load_fixture_pairs, resolve_fixture_path
from ab_test.material_database import get_material_by_id, search_materials
from ab_test.spectrum_database import get_data_by_id, get_datasets


def test_fixture_pair_manifest_files_exist() -> None:
    pairs = load_fixture_pairs()

    assert {pair["formula"] for pair in pairs} >= {"CuO", "NiO", "ZnO"}
    for pair in pairs:
        assert resolve_fixture_path(pair["cif_path"]).exists()
        assert resolve_fixture_path(pair["xas_path"]).exists()


def test_fixture_databases_are_backed_by_manifest() -> None:
    nio_pair = get_fixture_pair("NiO")
    assert nio_pair is not None

    assert search_materials("NiO") == "mp-19009"
    assert get_material_by_id("mp-19009") is not None

    datasets = get_datasets()
    dataset_ids = {dataset_id for dataset_id, _formula in datasets.values()}
    assert nio_pair["id"] in dataset_ids
    assert get_data_by_id(nio_pair["id"]) == [str(resolve_fixture_path(nio_pair["xas_path"]))]
