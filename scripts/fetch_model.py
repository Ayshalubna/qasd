"""Download the published model (built by the CI training job) into export/:  python -m scripts.fetch_model"""
import json
import urllib.request

from qasd.config import EXPORT

BASE = "https://raw.githubusercontent.com/Ayshalubna/qasd/model/export/"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "results.json", "export_check.json", "onnx/model_quantized.onnx"]


def main() -> None:
    for f in FILES:
        dest = EXPORT / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE + f, timeout=120) as r:
            dest.write_bytes(r.read())
        print("fetched", f)
    print(json.loads((EXPORT / "export_check.json").read_text()))


if __name__ == "__main__":
    main()
