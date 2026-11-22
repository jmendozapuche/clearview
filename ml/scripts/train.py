"""Train the ClearView detector (Ultralytics YOLO11n by default: small enough for Quest 3).

    python ml/scripts/train.py --data datasets/clearview/data.yaml --epochs 100
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml produced by prepare_dataset.py")
    ap.add_argument("--model", default="yolo11n.pt", help="starting weights (COCO-pretrained n/s recommended)")
    ap.add_argument("--imgsz", type=int, default=416, help="training size; export uses 320x416")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=float, default=32, help="-1 = let Ultralytics pick for the GPU")
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU, or cpu")
    ap.add_argument("--project", default="runs/clearview")
    ap.add_argument("--name", default="train")
    ap.add_argument("--resume", action="store_true", help="continue <project>/<name>/weights/last.pt if it exists")
    args = ap.parse_args()

    from pathlib import Path

    from ultralytics import YOLO

    last = Path(args.project) / args.name / "weights" / "last.pt"
    if args.resume and last.exists():
        print(f"Resuming from {last}")
        YOLO(str(last)).train(resume=True)
        return
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch,
        exist_ok=True,
        device=args.device,
        project=args.project,
        name=args.name,
        # Passthrough camera frames are noisy and often low light: lean on colour/brightness augmentation.
        hsv_v=0.5,
        hsv_s=0.6,
        degrees=5.0,
        perspective=0.0005,  # billboards are seen at oblique angles
        close_mosaic=10,
    )
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print("Per-class mAP50-95 (check that SAFETY classes have high recall, not just ads):")
    for i, name in model.names.items():
        try:
            print(f"  {name:<16} {metrics.box.maps[i]:.3f}")
        except (IndexError, AttributeError):
            pass


if __name__ == "__main__":
    main()
