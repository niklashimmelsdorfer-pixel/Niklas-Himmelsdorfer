#!/usr/bin/env python3
"""Baut aus index.html + styles.css + app.js eine einzelne, eigenständige Datei.

  python3 tools/build.py                -> proportio.html (per Doppelklick öffnen)
  python3 tools/build.py --artifact DIR -> Seite für gehostete Sandbox (Modell/WASM lokal)
"""
import re, sys, base64, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
html = (root / "index.html").read_text()
css = (root / "styles.css").read_text()
js = (root / "app.js").read_text()
sample = "data:image/jpeg;base64," + base64.b64encode((root / "assets" / "sample-front.jpg").read_bytes()).decode()

def inline(html, config=""):
    html = html.replace('<link rel="stylesheet" href="styles.css" />', f"<style>\n{css}</style>")
    cfg = f"<script>window.PROPORTIO_CONFIG = {config};</script>\n  " if config else ""
    return html.replace('<script type="module" src="app.js"></script>', f'{cfg}<script type="module">\n{js}</script>')

if "--artifact" in sys.argv:
    out = pathlib.Path(sys.argv[sys.argv.index("--artifact") + 1])
    out.mkdir(parents=True, exist_ok=True)
    page = inline(html, '{ wasmBinaryUrl: "vision_wasm_internal.wasm", modelUrl: "face_landmarker_model.wasm", noCamera: true, sampleUrl: "%s" }' % sample)
    head = re.search(r"<head>(.*?)</head>", page, re.S).group(1)
    body = re.search(r"<body>(.*)</body>", page, re.S).group(1)
    head = re.sub(r'\s*<meta charset[^>]*>\s*<meta name="viewport"[^>]*>', "", head)
    (out / "index.html").write_text(head.strip() + "\n" + body)
    print("wrote", out / "index.html")
else:
    (root / "proportio.html").write_text(inline(html, '{ sampleUrl: "%s" }' % sample))
    print("wrote proportio.html")
