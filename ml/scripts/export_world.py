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
import json
from pathlib import Path

AD_CLASSES = ["billboard", "advertisement poster", "logo", "shop sign", "advertising screen"]
SAFETY_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck",
                  "traffic light", "stop sign", "traffic sign"]
CHUNK_BYTES = 12 * 1024 * 1024


def fp16_weight_storage(src: Path, dst: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))
    g = m.graph
    inits, casts = [], []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= 1024:
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            inits.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT,
                                          name=init.name + "_cast"))
        else:
            inits.append(init)
    del g.initializer[:]
    g.initializer.extend(inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def decompose_softmax(path: Path) -> int:
    """Replace every Softmax with explicit ops: exp(x - max(x)) / sum(exp(x - max(x))) on the same axis.

    Mathematically identical. Needed because onnxruntime-web 1.18's WebGPU Softmax only supports the
    last axis (YOLO's DFL head uses axis 1). Simply transposing around the Softmax does not work: ORT's
    transpose optimizer folds the transposes back at load time. Returns the number of nodes replaced.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(path))
    opset = next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
    new_nodes, count = [], 0
    for n in m.graph.node:
        if n.op_type != "Softmax":
            new_nodes.append(n)
            continue
        if opset < 13:
            raise RuntimeError("decompose_softmax expects opset >= 13 (Softmax semantics changed there)")
        axis = next((helper.get_attribute_value(a) for a in n.attribute if a.name == "axis"), -1)
        p = n.name.strip("/").replace("/", "_") + "_sm"
        x, y = n.input[0], n.output[0]
        axes_name = p + "_axes"
        m.graph.initializer.append(numpy_helper.from_array(np.array([axis], dtype=np.int64), axes_name))
        reduce_max = (helper.make_node("ReduceMax", [x], [p + "_max"], axes=[axis], keepdims=1, name=p + "_max")
                      if opset < 18 else
                      helper.make_node("ReduceMax", [x, axes_name], [p + "_max"], keepdims=1, name=p + "_max"))
        new_nodes += [
            reduce_max,
            helper.make_node("Sub", [x, p + "_max"], [p + "_shift"], name=p + "_shift"),
            helper.make_node("Exp", [p + "_shift"], [p + "_exp"], name=p + "_exp"),
            helper.make_node("ReduceSum", [p + "_exp", axes_name], [p + "_sum"], keepdims=1, name=p + "_sum"),
            helper.make_node("Div", [p + "_exp", p + "_sum"], [y], name=p + "_div"),
        ]
        count += 1
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return count


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

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = out / "clearview-world.onnx"
    if args.fp32:
        final.write_bytes(onnx_path.read_bytes())
    else:
        fp16_weight_storage(onnx_path, final)
    print(f"Softmax nodes decomposed for WebGPU compatibility: {decompose_softmax(final)}")

    labels = ["# index,name,category  (YOLO-World text prompts)"]
    labels += [f"{i},{n},{'ad' if n in AD_CLASSES else 'safety'}" for i, n in enumerate(classes)]
    (out / "clearview_labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    manifest = {"input": [1, 3, args.height, args.width], "classes": classes,
                "categories": ["ad" if n in AD_CLASSES else "safety" for n in classes], "chunks": []}
    if not args.no_split:
        data = final.read_bytes()
        for i in range(0, len(data), CHUNK_BYTES):
            name = f"clearview-world.part{i // CHUNK_BYTES}"
            (out / name).write_bytes(data[i:i + CHUNK_BYTES])
            manifest["chunks"].append(name)
        manifest["bytes"] = len(data)
        final.unlink()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote model to {out}: {manifest['chunks'] or final.name}")


if __name__ == "__main__":
    main()
