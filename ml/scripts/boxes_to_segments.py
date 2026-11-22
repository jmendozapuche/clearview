"""Turn every box label of a YOLO dataset into a polygon with SAM, so a segmentation model can train on it.

Open Images, LogoDet-3K and the safety pseudo-labels only have boxes; your own photos from
web/etiquetar.html already have polygons, which are kept untouched. Ultralytics drops all polygons
of a dataset that mixes boxes and polygons, so after this script every label is a polygon.

    python ml/scripts/boxes_to_segments.py --data datasets/clearview --model sam2.1_b.pt

If SAM's mask is implausible for a box (almost empty, or mostly outside it), the box itself is
used as a rectangle, so no object is ever lost.
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image(label: Path, images_root: Path, labels_root: Path) -> Path | None:
    rel = label.relative_to(labels_root).with_suffix("")
    for ext in IMAGE_EXTS:
        p = images_root / rel.parent / (rel.name + ext)
        if p.exists():
            return p
    return None


def rect_polygon(cx: float, cy: float, w: float, h: float) -> list[float]:
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def mask_to_polygon(mask, box_px, img_w: int, img_h: int, min_fill: float = 0.25):
    """Largest contour of the mask clipped to the box, normalized; None if implausible."""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = [int(round(v)) for v in box_px]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_w, x1), min(img_h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    m = np.zeros((img_h, img_w), np.uint8)
    m[y0:y1, x0:x1] = mask[y0:y1, x0:x1].astype(np.uint8)
    if m.sum() < min_fill * (x1 - x0) * (y1 - y0):
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(c) < 3:
        return None
    pts = []
    for x, y in c:
        pts += [float(x) / img_w, float(y) / img_h]
    return pts


def convert_file(label: Path, image: Path, sam, stats: dict) -> None:
    import numpy as np
    from PIL import Image

    lines = label.read_text(encoding="utf-8").splitlines()
    boxes = [(i, l.split()) for i, l in enumerate(lines) if len(l.split()) == 5]
    if not boxes:
        return
    w, h = Image.open(image).size
    xyxy = []
    for _, p in boxes:
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        xyxy.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    masks = None
    try:
        r = sam.predict(str(image), bboxes=xyxy, verbose=False)[0]
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy() > 0.5
    except Exception as err:  # noqa: BLE001 - a broken image must not stop a 40k-image run
        stats["errors"] += 1
        print(f"  SAM failed on {image.name}: {err}")
    for k, (i, p) in enumerate(boxes):
        poly = None
        if masks is not None and k < len(masks):
            mk = masks[k]
            if mk.shape != (h, w):
                from PIL import Image as _I
                mk = np.asarray(_I.fromarray(mk.astype(np.uint8) * 255).resize((w, h))) > 127
            poly = mask_to_polygon(mk, xyxy[k], w, h)
        if poly is None:
            poly = rect_polygon(*(float(v) for v in p[1:5]))
            stats["rect"] += 1
        else:
            stats["sam"] += 1
        lines[i] = p[0] + " " + " ".join(f"{min(1.0, max(0.0, v)):.6f}" for v in poly)
    label.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root with images/ and labels/ (prepare_dataset output)")
    ap.add_argument("--model", default="sam2.1_b.pt")
    ap.add_argument("--max-images", type=int)
    args = ap.parse_args()

    from ultralytics import SAM

    root = Path(args.data)
    images_root, labels_root = root / "images", root / "labels"
    sam = SAM(args.model)
    stats = {"files": 0, "sam": 0, "rect": 0, "errors": 0, "no_image": 0}
    labels = sorted(labels_root.rglob("*.txt"))
    for n, label in enumerate(labels):
        if args.max_images is not None and stats["files"] >= args.max_images:
            break
        image = find_image(label, images_root, labels_root)
        if image is None:
            stats["no_image"] += 1
            continue
        convert_file(label, image, sam, stats)
        stats["files"] += 1
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(labels)} labels  (SAM {stats['sam']}, rectangles {stats['rect']})")
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
