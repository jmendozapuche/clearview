"""Merge several YOLO-format datasets into one ClearView dataset with a unified class list.

Each source must contain an ``images/`` and a ``labels/`` folder (any sub-folders, Ultralytics layout).
Class names of every source are remapped through its ``map`` (``"*"`` = every class, ``null`` = drop).

    python ml/scripts/prepare_dataset.py --config ml/configs/datasets.yaml
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clearview_ml import IMAGE_EXTS, class_defs, load_config, read_source_names, unity_labels_text  # noqa: E402


def build_mapping(source_names: list[str], mapping: dict, target_index: dict[str, int]) -> dict[int, int]:
    """Source class id -> target class id. Unmapped classes are dropped."""
    out = {}
    wildcard = mapping.get("*")
    for i, name in enumerate(source_names):
        target = mapping.get(name, wildcard)
        if target is not None:
            out[i] = target_index[target]
    return out


def remap_label_lines(lines: list[str], id_map: dict[int, int]) -> list[str]:
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in id_map:
            continue
        # Keep every coordinate: 4 for a box, or a full polygon (segmentation labels).
        out.append(" ".join([str(id_map[cls])] + parts[1:]))
    return out


def find_pairs(src_root: Path):
    images_dir = src_root / "images"
    labels_dir = src_root / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} not found")
    for img in sorted(images_dir.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img.relative_to(images_dir)
        yield img, (labels_dir / rel).with_suffix(".txt"), rel


def place(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def prepare(cfg: dict, root: Path, copy: bool = False) -> dict:
    classes = class_defs(cfg)
    target_index = {c.name: i for i, c in enumerate(classes)}
    out_dir = (root / cfg.get("output", "datasets/clearview")).resolve()
    val_fraction = float(cfg.get("val_fraction", 0.1))
    rng = random.Random(cfg.get("seed", 0))
    stats = {"images": 0, "boxes": [0] * len(classes), "skipped_empty": 0}
    n_seen = 0

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = out_dir / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    for src in cfg.get("sources", []):
        src_root = root / src["path"]
        if src.get("optional") and not (src_root / "images").is_dir():
            print(f"(optional source {src['name']!r} not found at {src_root}, skipped)")
            continue
        id_map = build_mapping(read_source_names(src, root), src.get("map") or {}, target_index)
        keep_empty = bool(src.get("keep_images_without_boxes", False))
        max_images = src.get("max_images")
        pairs = list(find_pairs(src_root))
        rng.shuffle(pairs)
        used = 0
        for img, lbl, rel in pairs:
            if max_images is not None and used >= max_images:
                break
            lines = lbl.read_text(encoding="utf-8").splitlines() if lbl.exists() else []
            remapped = remap_label_lines(lines, id_map)
            if not remapped and not keep_empty:
                stats["skipped_empty"] += 1
                continue
            # Exact share instead of a coin flip per image: small sources never end up with an empty val set.
            n_seen += 1
            split = "val" if int(n_seen * val_fraction) > int((n_seen - 1) * val_fraction) else "train"
            stem = f"{src['name']}__" + "__".join(rel.with_suffix("").parts)
            place(img, out_dir / "images" / split / (stem + img.suffix.lower()), copy)
            (out_dir / "labels" / split / (stem + ".txt")).write_text(
                "\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8")
            for line in remapped:
                stats["boxes"][int(line.split()[0])] += 1
            stats["images"] += 1
            used += 1

    data_yaml = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: c.name for i, c in enumerate(classes)},
    }
    import yaml
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (out_dir / "clearview_labels.txt").write_text(unity_labels_text(classes), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=".", help="base directory for relative paths in the config")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stats = prepare(cfg, Path(args.root), copy=args.copy)
    print(f"images: {stats['images']}  (skipped without boxes: {stats['skipped_empty']})")
    for c, n in zip(class_defs(cfg), stats["boxes"]):
        warn = "   <-- no examples!" if n == 0 else ""
        print(f"  {c.name:<16} {c.category:<7} {n:>8}{warn}")


if __name__ == "__main__":
    main()
