"""Package an exported YOLO ONNX model for the ClearView web demo.

Shared by export_world.py (no-training YOLO-World) and export.py (trained model).
Produces <out>/manifest.json, clearview_labels.txt and the model split into <=12 MB chunks.
"""
from __future__ import annotations

import json
from pathlib import Path

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


def package_for_web(onnx_path: Path, classes: list[str], categories: list[str], out: Path,
                    input_hw: tuple[int, int], thresholds: dict, name: str = "clearview",
                    fp32: bool = False, split: bool = True) -> dict:
    """fp16 weight storage + WebGPU-safe Softmax + chunking + manifest. Returns the manifest."""
    if len(classes) != len(categories):
        raise ValueError("classes and categories must have the same length")
    if not any(c == "safety" for c in categories):
        raise ValueError("a ClearView model needs at least one safety class")
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"{name}.onnx"
    if fp32:
        final.write_bytes(Path(onnx_path).read_bytes())
    else:
        fp16_weight_storage(Path(onnx_path), final)
    decomposed = decompose_softmax(final)

    lines = ["# index,name,category"] + [f"{i},{c},{k}" for i, (c, k) in enumerate(zip(classes, categories))]
    (out / "clearview_labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {"name": name, "input": [1, 3, input_hw[0], input_hw[1]], "classes": classes,
                "categories": categories, "thresholds": thresholds, "chunks": []}
    if split:
        data = final.read_bytes()
        for i in range(0, len(data), CHUNK_BYTES):
            part = f"{name}.part{i // CHUNK_BYTES}"
            (out / part).write_bytes(data[i:i + CHUNK_BYTES])
            manifest["chunks"].append(part)
        manifest["bytes"] = len(data)
        final.unlink()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Web package in {out}: {manifest['chunks'] or final.name} (softmax decomposed: {decomposed})")
    return manifest
