"""Export trained weights to ONNX for Unity Inference Engine and copy them into the Unity folder.

    python ml/scripts/export.py --weights runs/clearview/train/weights/best.pt \
        --config ml/configs/datasets.yaml --unity unity/Assets/ClearView/Model

Smoke test without training (COCO model, TVs/monitors are treated as "ads"):

    python ml/scripts/export.py --weights yolo11n.pt --coco-smoke-test

Produces clearview.onnx (input [1,3,320,416], output [1, 4+classes, anchors], no NMS)
and clearview_labels.txt (class -> ad / safety / ignore) for ClearViewController.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clearview_ml import class_defs, load_config, unity_labels_text  # noqa: E402
from web_package import package_for_web  # noqa: E402


# Used only by --coco-smoke-test.
COCO_AD = {"tv"}
COCO_SAFETY = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
               "traffic light", "stop sign", "dog", "horse"}


def coco_smoke_labels(model_names: dict | list) -> str:
    names = [model_names[k] for k in sorted(model_names)] if isinstance(model_names, dict) else list(model_names)
    lines = ["# COCO smoke test: tv = ad, people/vehicles/traffic = safety"]
    for i, n in enumerate(names):
        cat = "ad" if n in COCO_AD else ("safety" if n in COCO_SAFETY else "ignore")
        lines.append(f"{i},{n.replace(',', ' ')},{cat}")
    return "\n".join(lines) + "\n"


def check_names(model_names: dict | list, cfg: dict) -> None:
    names = [model_names[k] for k in sorted(model_names)] if isinstance(model_names, dict) else list(model_names)
    expected = [c.name for c in class_defs(cfg)]
    if names != expected:
        raise SystemExit(f"Model classes {names} do not match config classes {expected}. "
                         "Export with the same config used for prepare_dataset.py.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", help="the datasets config used to build the dataset")
    ap.add_argument("--coco-smoke-test", action="store_true", help="export a COCO model with TV = ad (no training needed)")
    ap.add_argument("--unity", default="unity/Assets/ClearView/Model")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=416, help="416x320 ~ the 4:3 passthrough camera aspect")
    ap.add_argument("--opset", type=int, default=15)
    ap.add_argument("--web", help="also write the web-demo package (manifest + chunks) to this folder")
    ap.add_argument("--web-height", type=int, default=416, help="web input height (portrait: phones are held upright)")
    ap.add_argument("--web-width", type=int, default=320)
    ap.add_argument("--ad-threshold", type=float, default=0.45)
    ap.add_argument("--safety-threshold", type=float, default=0.20)
    args = ap.parse_args()

    from ultralytics import YOLO

    if not args.coco_smoke_test and not args.config:
        raise SystemExit("--config is required (or use --coco-smoke-test)")
    model = YOLO(args.weights)
    if args.coco_smoke_test:
        labels_text = coco_smoke_labels(model.names)
        n_classes = len(model.names)
    else:
        cfg = load_config(args.config)
        check_names(model.names, cfg)
        labels_text = unity_labels_text(class_defs(cfg))
        n_classes = len(cfg["classes"])

    onnx_path = Path(model.export(
        format="onnx", imgsz=(args.height, args.width), opset=args.opset,
        simplify=True, dynamic=False, half=False, nms=False))

    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3, args.height, args.width), np.float32)})[0]
        if out.ndim != 3 or out.shape[1] != 4 + n_classes:
            raise SystemExit(f"Unexpected output shape {out.shape}; expected (1, {4 + n_classes}, anchors)")
        print(f"ONNX check OK: output {out.shape}")
    except ImportError:
        print("onnxruntime not installed; skipping output-shape check")

    dest = Path(args.unity)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, dest / "clearview.onnx")
    (dest / "clearview_labels.txt").write_text(labels_text, encoding="utf-8")
    print(f"Copied model and labels to {dest}")
    print(f"In Unity set ClearViewController: inputWidth={args.width}, inputHeight={args.height}")

    if args.web:
        if args.coco_smoke_test:
            raise SystemExit("--web is for trained ClearView models (use export_world.py for a no-training web model)")
        web_onnx = Path(model.export(format="onnx", imgsz=(args.web_height, args.web_width), opset=args.opset,
                                     simplify=True, dynamic=False, half=False, nms=False))
        defs = class_defs(cfg)
        package_for_web(web_onnx, [c.name for c in defs], [c.category for c in defs], Path(args.web),
                        (args.web_height, args.web_width),
                        {"ad": args.ad_threshold, "safety": args.safety_threshold}, name="clearview-trained")


if __name__ == "__main__":
    main()
