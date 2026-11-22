"""Build a no-training ad detector from YOLO-World (open vocabulary) for the web demo or Unity.

The classes are given as text, encoded once with CLIP, and baked into the ONNX file, so the exported
model runs without CLIP. Ads vs safety categories match ClearView's label format.

    python ml/scripts/export_world.py --out web/model            # web demo: fp16 weights, split in chunks
    python ml/scripts/export_world.py --out unity/Assets/ClearView/Model --fp32 --no-split   # Unity

Why not int8: dynamic int8 quantisation was tested and lost most "person" detections, which is the
one thing ClearView must not miss. fp16 *storage* (weights cast back to fp32 at load) halves the size
with identical detections on the test photos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web_package import package_for_web  # noqa: E402

AD_CLASSES = ["billboard", "advertisement poster", "logo", "shop sign", "advertising screen"]
SAFETY_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck",
                  "traffic light", "stop sign", "traffic sign"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolov8s-worldv2.pt")
    ap.add_argument("--out", default="web/model")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=416)
    ap.add_argument("--fp32", action="store_true", help="keep full fp32 weights (larger file)")
    ap.add_argument("--no-split", action="store_true", help="write one .onnx file instead of chunks")
    args = ap.parse_args()

    from ultralytics import YOLO

    classes = AD_CLASSES + SAFETY_CLASSES
    model = YOLO(args.weights)
    model.set_classes(classes)
    onnx_path = Path(model.export(format="onnx", imgsz=(args.height, args.width), opset=15,
                                  simplify=True, dynamic=False, nms=False))
    # YOLO-World scores run lower than a trained detector: thresholds tuned on test photos.
    package_for_web(onnx_path, classes, ["ad" if n in AD_CLASSES else "safety" for n in classes],
                    Path(args.out), (args.height, args.width), {"ad": 0.25, "safety": 0.12},
                    name="clearview-world", fp32=args.fp32, split=not args.no_split)


if __name__ == "__main__":
    main()
