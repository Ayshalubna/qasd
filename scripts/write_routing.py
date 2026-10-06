"""Write web/routing.json (intent -> [team, urgency]) from qasd/routing.py:  python -m scripts.write_routing"""
import json

from qasd.config import ROOT
from qasd.routing import ROUTE


def main() -> None:
    (ROOT / "web" / "routing.json").write_text(json.dumps({k: list(v) for k, v in sorted(ROUTE.items())}, indent=0))


if __name__ == "__main__":
    main()
