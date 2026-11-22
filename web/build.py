"""Assemble the web demo into web/dist (one HTML page + model chunks + onnxruntime wasm + sample photo).

    python ml/scripts/export_world.py --out web/model     # once: builds the detector (needs ultralytics)
    python web/build.py                                    # Claude artifact version (web/dist)
    python web/build.py --standalone --webgpu --out web/dist-pages   # self-hosted version with GPU

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
# Same skeleton Claude artifacts wrap pages in, for self-hosting (e.g. GitHub Pages).
SKELETON_HEAD = (
    '<!doctype html><html lang="es"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
    "<style>:root{padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}"
    "body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>"
)
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--webgpu", action="store_true",
                    help="also ship the WebGPU runtime (18 MB wasm; for self-hosting, e.g. GitHub Pages)")
    ap.add_argument("--standalone", action="store_true",
                    help="wrap the page in its own <!doctype html> skeleton (Claude artifacts add one themselves)")
    ap.add_argument("--out", default=str(WEB / "dist"))
    args = ap.parse_args()
    out = Path(args.out)
    model_dir = WEB / "model"
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    (out / "ort").mkdir()

    html = (WEB / "src" / "index.template.html").read_text(encoding="utf-8")
    html = html.replace("/*__CORE__*/", (WEB / "src" / "core.js").read_text(encoding="utf-8"))
    html = html.replace("/*__APP__*/", (WEB / "src" / "app.js").read_text(encoding="utf-8"))
    if args.standalone:
        html = (SKELETON_HEAD + html + "</body></html>\n")
    (out / "index.html").write_text(html, encoding="utf-8")

    # Chunks are published with a .wasm name: artifact hosting only serves known web types, and
    # .wasm is a binary type that reaches fetch() byte-for-byte. The page reads them as raw bytes.
    served = []
    for name in manifest["chunks"]:
        shutil.copy2(model_dir / name, out / "model" / (name + ".wasm"))
        served.append(name + ".wasm")
    # Version = hash of the model bytes, used by the page to bypass stale browser/CDN caches.
    import hashlib
    h = hashlib.sha256()
    for name in manifest["chunks"]:
        h.update((model_dir / name).read_bytes())
    (out / "model" / "manifest.json").write_text(
        json.dumps({**manifest, "chunks": served, "webgpu": args.webgpu, "version": h.hexdigest()[:12]}, indent=2))
    for name in WASM_FILES + (["ort-wasm-simd.jsep.wasm"] if args.webgpu else []):
        shutil.copy2(ort_dist() / name, out / "ort" / name)
    shutil.copy2(WEB / "assets" / "sample.jpg", out / "sample.jpg")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"Built {out} ({total / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
