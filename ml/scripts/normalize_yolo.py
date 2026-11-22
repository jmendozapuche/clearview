"""Collect any YOLO-format export (Roboflow's train/valid/test folders, or images/ + labels/) into the
layout prepare_dataset.py expects, optionally collapsing every class into one (e.g. "logo").

    python ml/scripts/normalize_yolo.py --src raw/roboflow_logos --out raw/logos --single-class logo
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_for(img: Path) -> Path:
    parts = list(img.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_names(src: Path) -> list[str]:
    import yaml
    for y in sorted(src.rglob("data.yaml")):
        names = yaml.safe_load(y.read_text(encoding="utf-8")).get("names")
        if names:
            return [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)
    return []


def normalize(src: Path, out: Path, single_class: str | None, max_images: int | None = None) -> dict:
    names = [single_class] if single_class else read_names(src)
    (out / "images" / "all").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "all").mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for img in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS and "images" in p.parts):
        if max_images is not None and n_img >= max_images:
            break
        lbl = label_for(img)
        if not lbl.exists():
            continue
        lines = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:  # boxes only (polygon/segmentation rows are skipped)
                lines.append(" ".join(["0" if single_class else p[0]] + p[1:]))
        if not lines:
            continue
        stem = "__".join(img.relative_to(src).with_suffix("").parts).replace(" ", "_")
        shutil.copy2(img, out / "images" / "all" / (stem + img.suffix.lower()))
        (out / "labels" / "all" / (stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_img += 1
        n_box += len(lines)
    if not names:
        raise SystemExit("No data.yaml with class names found; pass --single-class")
    import yaml
    (out / "data.yaml").write_text(yaml.safe_dump({"names": names}, sort_keys=False), encoding="utf-8")
    return {"images": n_img, "boxes": n_box, "names": names}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--single-class")
    ap.add_argument("--max-images", type=int)
    args = ap.parse_args()
    s = normalize(Path(args.src), Path(args.out), args.single_class, args.max_images)
    print(f"{s['images']} images, {s['boxes']} boxes; classes: {s['names'][:10]}")


if __name__ == "__main__":
    main()
