"""Offline preview of the "blank panel" look: what ClearView aims for, with no speed limit.

Pipeline (big models, seconds per image, NOT real time):
  1. Ads: YOLO-World-X with many text prompts.
  2. Exact ad shapes: SAM 2.1 segments each detected box.
  3. Safety: YOLO11x-seg silhouettes of people, vehicles and traffic signals, slightly dilated.
  4. Paint each ad as a matte blank panel lit like the scene, and keep every safety silhouette
     in front of it (the panel is cut around people instead of being cancelled).

    python ml/scripts/preview_target.py --image web/assets/sample.jpg --out preview.jpg
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageFilter

AD_PROMPTS = ["billboard", "advertisement", "poster", "logo", "brand logo", "digital billboard",
              "led advertising screen", "shop sign", "banner", "neon sign"]
SAFETY_COCO = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"}


def detect_ads(path: str, conf: float, imgsz: int):
    from ultralytics import YOLO
    m = YOLO("yolov8x-worldv2.pt")
    m.set_classes(AD_PROMPTS)
    r = m.predict(path, conf=conf, imgsz=imgsz, agnostic_nms=True, iou=0.5, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), [AD_PROMPTS[int(c)] for c in r.boxes.cls]


def segment_boxes(path: str, boxes: np.ndarray, shape) -> list[np.ndarray]:
    from ultralytics import SAM
    if len(boxes) == 0:
        return []
    r = SAM("sam2.1_b.pt").predict(path, bboxes=boxes.tolist(), verbose=False)[0]
    masks = r.masks.data.cpu().numpy().astype(bool)
    out = []
    for box, mk in zip(boxes, masks):
        x0, y0, x1, y1 = box.astype(int)
        rect = np.zeros(shape, bool)
        rect[y0:y1, x0:x1] = True
        inside = mk & rect
        # Ads are mostly flat rectangles; if SAM grabbed something odd, fall back to the box.
        fill = inside.sum() / max(1, rect.sum())
        out.append(inside if 0.5 <= fill <= 1.0 else rect)
    return out


def safety_mask(path: str, shape, imgsz: int) -> np.ndarray:
    from ultralytics import YOLO
    m = YOLO("yolo11x-seg.pt")
    r = m.predict(path, conf=0.2, imgsz=imgsz, retina_masks=True, verbose=False)[0]
    mask = np.zeros(shape, bool)
    if r.masks is None:
        return mask
    for cls, mk in zip(r.boxes.cls.tolist(), r.masks.data.cpu().numpy()):
        if m.names[int(cls)] in SAFETY_COCO:
            mask |= mk.astype(bool)[: shape[0], : shape[1]]
    # Dilate a few pixels so the panel never touches a person's outline.
    img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))
    return np.asarray(img) > 0


def paint(img: np.ndarray, ad_masks: list[np.ndarray], keep: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    lum = img.astype(np.float32).mean(axis=2) / 255.0
    for mk in ad_masks:
        region = mk & ~keep
        if region.sum() < 50:
            continue
        ys, xs = np.nonzero(mk)
        y0, y1 = ys.min(), ys.max() + 1
        # Light the panel like its surroundings: brightness from a ring around the ad.
        ring = np.zeros_like(mk)
        ring[max(0, y0 - 15): y1 + 15, max(0, xs.min() - 15): xs.max() + 16] = True
        ring &= ~mk
        level = float(np.median(lum[ring])) if ring.any() else 0.5
        base = np.array([196, 197, 194], np.float32) * (0.55 + 0.6 * level)
        # Soft vertical gradient: a matte panel catches more light at the top.
        grad = np.linspace(1.06, 0.94, y1 - y0, dtype=np.float32)[:, None, None]
        panel = np.clip(base[None, None, :] * grad, 0, 255)
        sub = region[y0:y1]
        out[y0:y1][sub] = np.broadcast_to(panel, (y1 - y0, img.shape[1], 3))[sub]
        # Thin darker frame along the panel edge.
        edge = mk & ~np.asarray(Image.fromarray((mk * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))).astype(bool)
        edge &= ~keep
        out[edge] *= 0.7
    # Anti-alias: blend a slightly blurred version only at mask borders.
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    img = np.asarray(Image.open(args.image).convert("RGB"))
    shape = img.shape[:2]
    boxes, confs, labels = detect_ads(args.image, args.conf, args.imgsz)
    big = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0.5 * shape[0] * shape[1]
    boxes, confs = boxes[~big], confs[~big]
    labels = [l for l, b in zip(labels, big) if not b]
    print(f"{len(boxes)} ads: " + ", ".join(f"{l} {c:.2f}" for l, c in zip(labels, confs)))
    masks = segment_boxes(args.image, boxes, shape)
    keep = safety_mask(args.image, shape, args.imgsz)
    result = paint(img, masks, keep)
    side = np.concatenate([img, result], axis=1)
    Image.fromarray(side).save(args.out, quality=92)
    print("saved", args.out)


if __name__ == "__main__":
    main()
