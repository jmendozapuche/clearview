"""Assemble the web demo into web/dist (one HTML page + model chunks + onnxruntime wasm + sample photo).

    python ml/scripts/export_world.py --out web/model     # once: builds the detector (needs ultralytics)
    python web/build.py                                    # downloads onnxruntime-web via npm if needed

Then serve web/dist over HTTPS (live camera needs HTTPS) or publish it as a Claude artifact.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ORT_VERSION = "1.18.0"
WASM_FILES = ["ort-wasm-simd.wasm", "ort-wasm.wasm"]  # single-threaded builds (no cross-origin isolation needed)

WEB = Path(__file__).resolve().parent
DIST = WEB / "dist"
CACHE = WEB / ".cache"


def ort_dist() -> Path:
    pkg = CACHE / f"onnxruntime-web-{ORT_VERSION}" / "package" / "dist"
    if not pkg.exists():
        CACHE.mkdir(exist_ok=True)
        tgz = subprocess.run(["npm", "pack", f"onnxruntime-web@{ORT_VERSION}", "--silent"], cwd=CACHE,
                             check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]
        with tarfile.open(CACHE / tgz) as t:
            t.extractall(CACHE / f"onnxruntime-web-{ORT_VERSION}", filter="data")
    return pkg


def main() -> None:
    model_dir = WEB / "model"
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "model").mkdir(parents=True)
    (DIST / "ort").mkdir()

    html = (WEB / "src" / "index.template.html").read_text(encoding="utf-8")
    html = html.replace("/*__CORE__*/", (WEB / "src" / "core.js").read_text(encoding="utf-8"))
    html = html.replace("/*__APP__*/", (WEB / "src" / "app.js").read_text(encoding="utf-8"))
    (DIST / "index.html").write_text(html, encoding="utf-8")

    # Chunks are published with a .wasm name: artifact hosting only serves known web types, and
    # .wasm is a binary type that reaches fetch() byte-for-byte. The page reads them as raw bytes.
    served = []
    for name in manifest["chunks"]:
        shutil.copy2(model_dir / name, DIST / "model" / (name + ".wasm"))
        served.append(name + ".wasm")
    (DIST / "model" / "manifest.json").write_text(json.dumps({**manifest, "chunks": served}, indent=2))
    for name in WASM_FILES:
        shutil.copy2(ort_dist() / name, DIST / "ort" / name)
    shutil.copy2(WEB / "assets" / "sample.jpg", DIST / "sample.jpg")
    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"Built {DIST} ({total / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
