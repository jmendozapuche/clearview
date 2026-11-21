import sys
from pathlib import Path

import pytest

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "scripts"))

from clearview_ml import unity_labels_text, class_defs, validate_config  # noqa: E402
from prepare_dataset import build_mapping, prepare, remap_label_lines  # noqa: E402

CLASSES = [
    {"name": "logo", "category": "ad"},
    {"name": "billboard", "category": "ad"},
    {"name": "person", "category": "safety"},
]


def make_source(root: Path, name: str, labels: dict[str, str], names: list[str]):
    src = root / name
    for stem, text in labels.items():
        img = src / "images" / "train" / f"{stem}.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"fake")
        lbl = src / "labels" / "train" / f"{stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text(text)
    (src / "data.yaml").write_text("names: [" + ", ".join(names) + "]\n")


def test_mapping_with_wildcard_and_explicit_override():
    idx = {"logo": 0, "billboard": 1, "person": 2}
    m = build_mapping(["nike", "adidas", "person"], {"*": "logo", "person": "person"}, idx)
    assert m == {0: 0, 1: 0, 2: 2}


def test_mapping_drops_unmapped_and_null():
    idx = {"logo": 0, "billboard": 1, "person": 2}
    m = build_mapping(["cat", "person", "sign"], {"person": "person", "sign": None}, idx)
    assert m == {1: 2}


def test_remap_skips_dropped_classes_and_bad_lines():
    out = remap_label_lines(["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1", "garbage", "3 0.1 0.1 0.1 0.1"], {0: 2, 3: 0})
    assert out == ["2 0.5 0.5 0.1 0.1", "0 0.1 0.1 0.1 0.1"]


def test_prepare_merges_sources(tmp_path):
    make_source(tmp_path, "coco", {"a": "0 0.5 0.5 0.2 0.2\n", "b": "1 0.5 0.5 0.2 0.2\n"}, ["person", "cat"])
    make_source(tmp_path, "logos", {"c": "0 0.3 0.3 0.1 0.1\n1 0.6 0.6 0.1 0.1\n"}, ["nike", "adidas"])
    cfg = {
        "output": "out",
        "val_fraction": 0.0,
        "classes": CLASSES,
        "sources": [
            {"name": "coco", "path": "coco", "map": {"person": "person"}},
            {"name": "logos", "path": "logos", "map": {"*": "logo"}},
        ],
    }
    validate_config(cfg)
    stats = prepare(cfg, tmp_path, copy=True)
    assert stats["images"] == 2          # coco/b only had a cat -> skipped
    assert stats["skipped_empty"] == 1
    assert stats["boxes"] == [2, 0, 1]
    out = tmp_path / "out"
    assert (out / "labels" / "train" / "logos__train__c.txt").read_text().splitlines() == [
        "0 0.3 0.3 0.1 0.1", "0 0.6 0.6 0.1 0.1"]
    assert (out / "images" / "train" / "coco__train__a.jpg").exists()
    assert "2,person,safety" in (out / "clearview_labels.txt").read_text()
    assert "names:" in (out / "data.yaml").read_text()


def test_config_requires_a_safety_class():
    with pytest.raises(ValueError, match="safety"):
        validate_config({"classes": [{"name": "logo", "category": "ad"}]})


def test_config_rejects_unknown_target():
    with pytest.raises(ValueError, match="unknown class"):
        validate_config({"classes": CLASSES, "sources": [{"name": "x", "path": "x", "map": {"a": "nope"}}]})


def test_unity_labels_format_matches_csharp_parser():
    text = unity_labels_text(class_defs({"classes": CLASSES}))
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    assert lines == ["0,logo,ad", "1,billboard,ad", "2,person,safety"]


def test_example_config_is_valid():
    from clearview_ml import load_config
    load_config(ML / "configs" / "datasets.example.yaml")
