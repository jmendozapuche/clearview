"""Add safety boxes (people, vehicles, traffic lights...) to ad datasets using a COCO-pretrained model.

Why: logo/billboard datasets do not label the people in their photos. Merged as-is, the model would
learn that a person in a street photo is "background" -> weaker safety recall. Run this on every
ad source BEFORE prepare_dataset.py. Existing label lines are kept; COCO boxes are appended.

    python ml/scripts/pseudo_label_safety.py --source raw/billboards --model yolo11x.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clearview_ml import IMAGE_EXTS  # noqa: E402

# COCO classes copied into the source's own label space (appended to its data.yaml names).
DEFAULT_COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]


def iou_xywh(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def merge_boxes(existing: list[str], new: list[tuple[int, list[float]]], same: dict[int, set[int]],
                iou_threshold: float = 0.5) -> list[str]:
    """Append new boxes unless an existing box of an equivalent class already covers them."""
    old = []
    for line in existing:
        p = line.split()
        if len(p) == 5:
            old.append((int(float(p[0])), [float(v) for v in p[1:5]]))
        elif len(p) >= 7:  # polygon "cls x1 y1 x2 y2 ...": compare using its bounding box
            xs, ys = [float(v) for v in p[1::2]], [float(v) for v in p[2::2]]
            old.append((int(float(p[0])), [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                                           max(xs) - min(xs), max(ys) - min(ys)]))
    added = []
    for cls, xywh in new:
        if any(o_cls in same.get(cls, {cls}) and iou_xywh(o, xywh) > iou_threshold for o_cls, o in old):
            continue
        added.append(f"{cls} " + " ".join(f"{v:.6f}" for v in xywh))
        old.append((cls, xywh))
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="YOLO source folder with images/, labels/ and data.yaml")
    ap.add_argument("--model", default="yolo11x.pt", help="COCO model; bigger = better pseudo-labels")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    import yaml
    from ultralytics import YOLO

    src = Path(args.source)
    data_yaml = src / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    names = [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)

    # Reuse an existing class when the source already has it under ClearView's spelling
    # (e.g. "traffic_light" from Open Images), otherwise extend the label space (idempotent).
    for n in DEFAULT_COCO_CLASSES:
        if n not in names and n.replace(" ", "_") not in names:
            names.append(n)
    data["names"] = names
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    src_id = {n: i for i, n in enumerate(names)}
    for n in DEFAULT_COCO_CLASSES:
        if n not in src_id:
            src_id[n] = src_id[n.replace(" ", "_")]
    same = {i: {i} for i in range(len(names))}

    model = YOLO(args.model)
    coco_names = model.names
    wanted = {i for i, n in coco_names.items() if n in DEFAULT_COCO_CLASSES}

    images = [p for p in sorted((src / "images").rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    added = 0
    for img in images:
        rel = img.relative_to(src / "images")
        lbl = (src / "labels" / rel).with_suffix(".txt")
        done_marker = lbl.with_suffix(".pseudo")
        if done_marker.exists():
            continue  # already processed; makes the script safe to re-run
        lines = lbl.read_text(encoding="utf-8").splitlines() if lbl.exists() else []
        result = model.predict(str(img), conf=args.conf, imgsz=args.imgsz, classes=sorted(wanted), verbose=False)[0]
        found = [(src_id[coco_names[int(c)]], xywhn)
                 for c, xywhn in zip(result.boxes.cls.tolist(), result.boxes.xywhn.tolist())]
        new = merge_boxes(lines, found, same)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("\n".join(lines + new) + ("\n" if lines or new else ""), encoding="utf-8")
        done_marker.touch()
        added += len(new)
    print(f"{len(images)} images processed, {added} safety boxes added. Map the new names in your datasets config.")


if __name__ == "__main__":
    main()
