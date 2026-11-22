"""Download a ClearView subset of Open Images V7 (no account or key needed) in YOLO format.

Gives billboards and posters (ads) plus people, vehicles and traffic signals (safety), with boxes
already drawn by people. Writes <out>/images/<split>/*.jpg, <out>/labels/<split>/*.txt and a
data.yaml whose names are ClearView class names, ready for prepare_dataset.py.

    python ml/scripts/fetch_openimages.py --split train --out raw/openimages --ads-per-class 4000 --safety-per-class 2500

Notes
- Open Images marks "depictions" (a person printed on a poster, a drawn car). By default they keep
  their label, so the detector treats them as real (fail-open: the poster stays visible).
  --drop-safety-depictions removes those boxes from safety classes so the model can learn to tell
  printed people from real ones. Only use it if you then check safety recall on real people.
- Box annotations are not guaranteed exhaustive for every object; run pseudo_label_safety.py on
  the result to add people/vehicles the annotators skipped.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLASSES_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
BOXES_URL = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

# Open Images display name -> ClearView class name.
DEFAULT_MAP = {
    "Billboard": "billboard",
    "Poster": "poster",
    "Person": "person", "Man": "person", "Woman": "person", "Boy": "person", "Girl": "person",
    "Wheelchair": "wheelchair",
    "Bicycle": "bicycle",
    "Car": "car", "Taxi": "car", "Van": "car",
    "Motorcycle": "motorcycle",
    "Bus": "bus",
    "Truck": "truck",
    "Traffic light": "traffic_light",
    "Stop sign": "stop_sign",
    "Traffic sign": "traffic_sign",
}
AD_TARGETS = {"billboard", "poster"}


def download(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)
    return dst


def read_mids(classes_csv: Path, mapping: dict[str, str]) -> dict[str, str]:
    """MID (/m/...) -> ClearView class name."""
    mids = {}
    with open(classes_csv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1].strip() in mapping:
                mids[row[0].strip()] = mapping[row[1].strip()]
    missing = set(mapping) - {k for k in mapping if mapping[k] in mids.values()}
    if missing:
        print(f"warning: not found in Open Images: {sorted(missing)}")
    return mids


def select_images(boxes_csv: Path, mids: dict[str, str], ads_cap: int, safety_cap: int, seed: int) -> set[str]:
    """Pick images so each ad class gets up to ads_cap images and each safety class up to safety_cap."""
    by_class: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    with open(boxes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target = mids.get(row["LabelName"])
            if target and row["ImageID"] not in seen[target]:
                seen[target].add(row["ImageID"])
                by_class[target].append(row["ImageID"])
    rng = random.Random(seed)
    chosen: set[str] = set()
    # Ads first: they are the rare classes, and their images also bring people and vehicles.
    order = sorted(by_class, key=lambda c: (c not in AD_TARGETS, len(by_class[c])))
    for cls in order:
        cap = ads_cap if cls in AD_TARGETS else safety_cap
        have = len(seen[cls] & chosen)
        pool = [i for i in by_class[cls] if i not in chosen]
        rng.shuffle(pool)
        take = pool[:max(0, cap - have)]
        chosen.update(take)
        print(f"  {cls:<14} available {len(by_class[cls]):>7}  in selection {have + len(take):>6}")
    return chosen


def collect_labels(boxes_csv: Path, mids: dict[str, str], names: list[str], chosen: set[str],
                   drop_safety_depictions: bool) -> dict[str, list[str]]:
    index = {n: i for i, n in enumerate(names)}
    labels: dict[str, list[str]] = defaultdict(list)
    with open(boxes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id = row["ImageID"]
            target = mids.get(row["LabelName"])
            if not target or image_id not in chosen:
                continue
            if drop_safety_depictions and target not in AD_TARGETS and row.get("IsDepiction") == "1":
                continue
            x0, x1 = float(row["XMin"]), float(row["XMax"])
            y0, y1 = float(row["YMin"]), float(row["YMax"])
            if x1 <= x0 or y1 <= y0:
                continue
            labels[image_id].append(
                f"{index[target]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
    return labels


def fetch_image(split: str, image_id: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    try:
        with urllib.request.urlopen(IMAGE_URL.format(split=split, image_id=image_id), timeout=30) as r:
            dst.write_bytes(r.read())
        return True
    except Exception:  # noqa: BLE001 - a few images are always missing; skip them
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=sorted(BOXES_URL))
    ap.add_argument("--out", default="raw/openimages")
    ap.add_argument("--cache", default="raw/openimages_cache", help="where annotation CSVs are stored")
    ap.add_argument("--ads-per-class", type=int, default=4000)
    ap.add_argument("--safety-per-class", type=int, default=2500)
    ap.add_argument("--drop-safety-depictions", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.cache)
    classes_csv = download(CLASSES_URL, cache / "classes.csv")
    boxes_csv = download(BOXES_URL[args.split], cache / f"{args.split}-bbox.csv")
    mids = read_mids(classes_csv, DEFAULT_MAP)
    names = sorted(set(DEFAULT_MAP.values()), key=list(dict.fromkeys(DEFAULT_MAP.values())).index)

    print("Selecting images:")
    chosen = select_images(boxes_csv, mids, args.ads_per_class, args.safety_per_class, args.seed)
    labels = collect_labels(boxes_csv, mids, names, chosen, args.drop_safety_depictions)

    out = Path(args.out)
    img_dir, lbl_dir = out / "images" / args.split, out / "labels" / args.split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(labels)
    print(f"Downloading {len(ids)} images with {args.workers} workers...")
    with ThreadPoolExecutor(args.workers) as pool:
        ok = list(pool.map(lambda i: fetch_image(args.split, i, img_dir / f"{i}.jpg"), ids))
    written = 0
    for image_id, good in zip(ids, ok):
        if good:
            (lbl_dir / f"{image_id}.txt").write_text("\n".join(labels[image_id]) + "\n", encoding="utf-8")
            written += 1
    import yaml
    (out / "data.yaml").write_text(yaml.safe_dump({"names": names}, sort_keys=False), encoding="utf-8")
    print(f"Done: {written} images ({len(ids) - written} unavailable) in {out}")


if __name__ == "__main__":
    sys.exit(main())
