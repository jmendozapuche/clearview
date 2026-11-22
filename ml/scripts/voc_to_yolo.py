"""Convert a Pascal VOC dataset (one .xml per image, any folder layout) to Ultralytics YOLO format.

Made for LogoDet-3K (Category/Brand/123.jpg + 123.xml) but works for any VOC export.
With --single-class every object becomes that class (e.g. every brand -> "logo").

    python ml/scripts/voc_to_yolo.py --src raw/LogoDet-3K --out raw/logos --single-class logo --max-images 8000
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_voc(xml_path: Path):
    """Returns (width, height, [(name, xmin, ymin, xmax, ymax)]), or None if unusable."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    try:
        w = float(size.find("width").text)
        h = float(size.find("height").text)
    except (AttributeError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    objs = []
    for o in root.iter("object"):
        bb = o.find("bndbox")
        name = (o.findtext("name") or "").strip()
        if bb is None or not name:
            continue
        try:
            x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            continue
        x0, x1 = max(0.0, min(x0, x1)), min(w, max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(h, max(y0, y1))
        if x1 - x0 >= 1 and y1 - y0 >= 1:
            objs.append((name, x0, y0, x1, y1))
    return w, h, objs


def find_image(xml_path: Path, root: Path, filename: str | None) -> Path | None:
    for ext in IMAGE_EXTS:
        p = xml_path.with_suffix(ext)
        if p.exists():
            return p
    if filename:
        for cand in (xml_path.parent / filename, root / filename, xml_path.parent.parent / "JPEGImages" / filename):
            if cand.exists():
                return cand
    return None


def convert(src: Path, out: Path, single_class: str | None, max_images: int | None, seed: int = 0) -> dict:
    xmls = sorted(src.rglob("*.xml"))
    random.Random(seed).shuffle(xmls)
    names: list[str] = [single_class] if single_class else []
    index = {n: i for i, n in enumerate(names)}
    (out / "images" / "all").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "all").mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes": 0, "skipped": 0}
    for xml_path in xmls:
        if max_images is not None and stats["images"] >= max_images:
            break
        parsed = parse_voc(xml_path)
        fname = ET.parse(xml_path).getroot().findtext("filename")
        img = find_image(xml_path, src, fname)
        if parsed is None or img is None or not parsed[2]:
            stats["skipped"] += 1
            continue
        w, h, objs = parsed
        lines = []
        for name, x0, y0, x1, y1 in objs:
            cls = single_class or name
            if cls not in index:
                index[cls] = len(names)
                names.append(cls)
            lines.append(f"{index[cls]} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        stem = "__".join(xml_path.relative_to(src).with_suffix("").parts).replace(" ", "_")
        shutil.copy2(img, out / "images" / "all" / (stem + img.suffix.lower()))
        (out / "labels" / "all" / (stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["images"] += 1
        stats["boxes"] += len(lines)
    import yaml
    (out / "data.yaml").write_text(yaml.safe_dump({"names": names}, sort_keys=False), encoding="utf-8")
    stats["names"] = names
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--single-class", help="map every object to this class name")
    ap.add_argument("--max-images", type=int)
    args = ap.parse_args()
    s = convert(Path(args.src), Path(args.out), args.single_class, args.max_images)
    print(f"{s['images']} images, {s['boxes']} boxes, {s['skipped']} skipped; classes: {s['names'][:10]}")


if __name__ == "__main__":
    main()
