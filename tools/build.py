#!/usr/bin/env python3
"""Build-Werkzeug für Proportio.

  python3 tools/build.py                 -> proportio.html (Einzeldatei, per Doppelklick öffnen)
  python3 tools/build.py --site DIR      -> öffentliche Website (für GitHub Pages / Netlify / Vercel)
  python3 tools/build.py --artifact DIR  -> Testversion für eine Sandbox mit lokal ausgeliefertem Modell
"""
import re, sys, base64, shutil, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
LEGAL = ["impressum.html", "datenschutz.html", "agb.html", "widerruf.html"]
html = (root / "index.html").read_text()
css = (root / "styles.css").read_text()
js = (root / "app.js").read_text()
config_js = (root / "config.js").read_text()
b64 = lambda p: base64.b64encode((root / p).read_bytes()).decode()
sample = "data:image/jpeg;base64," + b64("assets/sample-front.jpg")

# Schriften als data:-URIs, damit Einzeldateien ohne externe Schrift-Server auskommen
fonts_css = (root / "assets" / "fonts.css").read_text()
fonts_inline = re.sub(r'url\("fonts/([^"]+)"\)', lambda m: f'url("data:font/woff2;base64,{b64("assets/fonts/" + m.group(1))}")', fonts_css)
FONT_LINKS = re.compile(r'\s*<link rel="preload" href="assets/fonts/[^"]+"[^>]*>|\s*<link rel="stylesheet" href="assets/fonts.css" />')


def single_file(overrides: str) -> str:
    """index.html mit allem inline: Schriften, CSS, Konfiguration, App."""
    page = FONT_LINKS.sub("", html)
    page = page.replace('<link rel="stylesheet" href="styles.css" />', f"<style>\n{fonts_inline}\n{css}</style>")
    scripts = f"<script>window.PROPORTIO_CONFIG = {overrides};</script>\n  <script>\n{config_js}</script>\n  <script type=\"module\">\n{js}</script>"
    page = page.replace('<script src="config.js"></script>\n  <script type="module" src="app.js"></script>', scripts)
    assert "app.js" not in page.split("<script type=\"module\">")[0][-200:], "Skript-Einbindung nicht gefunden"
    return page


if "--site" in sys.argv:
    out = pathlib.Path(sys.argv[sys.argv.index("--site") + 1]).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in ["index.html", "styles.css", "app.js", "config.js", *LEGAL]:
        shutil.copy2(root / f, out / f)
    shutil.copytree(root / "assets", out / "assets")
    for extra in ["CNAME", "robots.txt"]:
        if (root / extra).exists():
            shutil.copy2(root / extra, out / extra)
    (out / ".nojekyll").write_text("")
    print("Website gebaut:", out)
elif "--artifact" in sys.argv:
    out = pathlib.Path(sys.argv[sys.argv.index("--artifact") + 1])
    out.mkdir(parents=True, exist_ok=True)
    page = single_file('{ wasmBinaryUrl: "vision_wasm_internal.wasm", modelUrl: "face_landmarker_model.wasm", noCamera: true, sampleUrl: "%s" }' % sample)
    head = re.search(r"<head>(.*?)</head>", page, re.S).group(1)
    body = re.search(r"<body>(.*)</body>", page, re.S).group(1)
    head = re.sub(r'\s*<meta charset[^>]*>\s*<meta name="viewport"[^>]*>', "", head)
    (out / "index.html").write_text(head.strip() + "\n" + body)
    for f in LEGAL:  # Rechtsseiten als eigenständige Seiten mit eingebettetem CSS
        p = (root / f).read_text()
        p = FONT_LINKS.sub("", p).replace('<link rel="stylesheet" href="styles.css" />', f"<style>\n{fonts_inline}\n{css}</style>")
        (out / f).write_text(p)
    print("Testversion gebaut:", out)
else:
    (root / "proportio.html").write_text(single_file('{ sampleUrl: "%s" }' % sample))
    print("wrote proportio.html")
