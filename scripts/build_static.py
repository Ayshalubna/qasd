"""Package the in-browser demo for a static host into dist/static/:  python -m scripts.build_static"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "static"
README = """---
title: Qasd — Banking Intent Routing
emoji: 🏦
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: true
short_description: LoRA-fine-tuned DistilBERT routes bank messages, in-browser
---

Live demo of [github.com/Ayshalubna/qasd](https://github.com/Ayshalubna/qasd). The int8 ONNX model runs in your
browser with transformers.js. Data: Banking77 (PolyAI, CC BY 4.0).
"""


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name in ("index.html", "app.js", "routing.json"):
        shutil.copy(ROOT / "web" / name, OUT / name)
    for f in (ROOT / "web" / "fonts").glob("*.woff2"):
        shutil.copy(f, OUT / f.name)
    (OUT / "styles.css").write_text((ROOT / "web" / "styles.css").read_text().replace("url(fonts/", "url("))
    (OUT / "README.md").write_text(README)
    print(f"built {OUT}: {len(list(OUT.iterdir()))} files")


if __name__ == "__main__":
    main()
