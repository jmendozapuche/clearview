"""Evaluate a trained ClearView detector, putting safety first.

Prints precision / recall / mAP50 per class and flags every SAFETY class whose recall is below the
bar: a missed person or car is the failure that can put someone at risk (a missed ad only means an
ad stays visible).

    python ml/scripts/eval_safety.py --weights runs/clearview/train/weights/best.pt \
        --data datasets/clearview/data.yaml --config ml/configs/colab.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clearview_ml import class_defs, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.20, help="the app's safety threshold")
    ap.add_argument("--min-safety-recall", type=float, default=0.80)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    cats = {c.name: c.category for c in class_defs(load_config(args.config))}
    model = YOLO(args.weights)
    m = model.val(data=args.data, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False, plots=False)

    present = list(getattr(m.box, "ap_class_index", []))
    print(f"\n{'clase':<15}{'tipo':<8}{'precisión':>10}{'recall':>8}{'mAP50':>8}")
    weak = []
    for name_idx, name in model.names.items():
        cat = cats.get(name, "?")
        if name_idx not in present:
            print(f"{name:<15}{cat:<8}{'sin ejemplos en validación':>26}")
            if cat == "safety":
                weak.append((name, None))
            continue
        p, r, ap50, _ = m.box.class_result(present.index(name_idx))
        flag = ""
        if cat == "safety" and r < args.min_safety_recall:
            flag = "  <-- recall bajo"
            weak.append((name, r))
        print(f"{name:<15}{cat:<8}{p:>10.2f}{r:>8.2f}{ap50:>8.2f}{flag}")

    if weak:
        print(f"\nAVISO: clases de seguridad por debajo de recall {args.min_safety_recall:.2f}: "
              + ", ".join(n if r is None else f"{n} ({r:.2f})" for n, r in weak))
        print("Con el umbral de la app, parte de estos objetos no se detectaría y un anuncio a su lado podría taparse.")
        print("Más datos de esas clases, más épocas o un umbral de seguridad más bajo (--safety-threshold en export.py).")
        return 1
    print("\nOK: todas las clases de seguridad superan el recall mínimo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
