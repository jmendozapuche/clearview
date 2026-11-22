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


# ---------- voc_to_yolo ----------
from voc_to_yolo import convert  # noqa: E402

VOC = """<annotation><filename>{f}</filename><size><width>200</width><height>100</height><depth>3</depth></size>
<object><name>{n}</name><bndbox><xmin>20</xmin><ymin>10</ymin><xmax>60</xmax><ymax>50</ymax></bndbox></object>
<object><name>{n}</name><bndbox><xmin>150</xmin><ymin>0</ymin><xmax>260</xmax><ymax>40</ymax></bndbox></object>
</annotation>"""


def test_voc_to_yolo_single_class_and_clipping(tmp_path):
    for brand in ("Dell", "Puma"):
        d = tmp_path / "src" / "Electronic" / brand
        d.mkdir(parents=True)
        (d / "1.jpg").write_bytes(b"x")
        (d / "1.xml").write_text(VOC.format(f="1.jpg", n=brand))
    (tmp_path / "src" / "orphan.xml").write_text(VOC.format(f="missing.jpg", n="X"))
    s = convert(tmp_path / "src", tmp_path / "out", "logo", None)
    assert s["images"] == 2 and s["skipped"] == 1 and s["names"] == ["logo"]
    lbl = (tmp_path / "out" / "labels" / "all" / "Electronic__Dell__1.txt").read_text().split("\n")
    assert lbl[0] == "0 0.200000 0.300000 0.200000 0.400000"
    # second box is clipped to the image: x 150..200
    assert lbl[1].startswith("0 0.875000 0.200000 0.250000 0.400000")


def test_voc_to_yolo_keeps_names_without_single_class(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.jpg").write_bytes(b"x")
    (d / "a.xml").write_text(VOC.format(f="a.jpg", n="Nike"))
    s = convert(d, tmp_path / "out", None, None)
    assert s["names"] == ["Nike"]


# ---------- pseudo_label_safety ----------
from pseudo_label_safety import merge_boxes  # noqa: E402


def test_merge_boxes_skips_duplicates_keeps_new():
    existing = ["2 0.5 0.5 0.2 0.4"]
    found = [(2, [0.51, 0.5, 0.2, 0.4]), (2, [0.1, 0.1, 0.05, 0.1]), (3, [0.5, 0.5, 0.2, 0.4])]
    added = merge_boxes(existing, found, {})
    assert len(added) == 2                       # duplicate person dropped
    assert added[0].startswith("2 0.100000")     # far person kept
    assert added[1].startswith("3 ")             # other class at the same spot kept


def test_optional_missing_source_is_skipped(tmp_path):
    make_source(tmp_path, "coco", {"a": "0 0.5 0.5 0.2 0.2\n"}, ["person"])
    cfg = {"output": "out", "val_fraction": 0.0, "classes": CLASSES,
           "sources": [{"name": "coco", "path": "coco", "map": {"person": "person"}},
                       {"name": "mine", "path": "nope", "optional": True, "map": {}}]}
    assert prepare(cfg, tmp_path, copy=True)["images"] == 1


def test_colab_config_is_valid():
    from clearview_ml import load_config
    load_config(ML / "configs" / "colab.yaml")


def test_val_split_is_exact(tmp_path):
    make_source(tmp_path, "s", {f"i{i}": "0 0.5 0.5 0.2 0.2\n" for i in range(20)}, ["person"])
    cfg = {"output": "out", "val_fraction": 0.1, "classes": CLASSES,
           "sources": [{"name": "s", "path": "s", "map": {"person": "person"}}]}
    prepare(cfg, tmp_path, copy=True)
    assert len(list((tmp_path / "out" / "images" / "val").iterdir())) == 2
    assert len(list((tmp_path / "out" / "images" / "train").iterdir())) == 18


# ---------- normalize_yolo ----------
from normalize_yolo import normalize  # noqa: E402


def test_normalize_roboflow_layout_single_class(tmp_path):
    src = tmp_path / "rf"
    for split in ("train", "valid"):
        (src / split / "images").mkdir(parents=True)
        (src / split / "labels").mkdir(parents=True)
        (src / split / "images" / "a.jpg").write_bytes(b"x")
        (src / split / "labels" / "a.txt").write_text("3 0.5 0.5 0.1 0.1\n1 0.1 0.1 0.1 0.1 0.2 0.2\n")
    (src / "data.yaml").write_text("names: [nike, puma, dell, apple]\n")
    s = normalize(src, tmp_path / "out", "logo")
    assert s == {"images": 2, "boxes": 2, "names": ["logo"]}
    assert (tmp_path / "out" / "labels" / "all" / "train__images__a.txt").read_text() == "0 0.5 0.5 0.1 0.1\n"


def test_normalize_keeps_names(tmp_path):
    src = tmp_path / "rf"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    (src / "images" / "b.png").write_bytes(b"x")
    (src / "labels" / "b.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    (src / "data.yaml").write_text("names: {0: billboard, 1: person}\n")
    assert normalize(src, tmp_path / "out", None)["names"] == ["billboard", "person"]
