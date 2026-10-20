"""Read-only source diagnostics, usable even when the preflight gate fails."""
from __future__ import annotations

import argparse
import json

from shared_weather_benchmark.provenance import inspect_sources, source_diagnostic_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print full machine-readable diagnostics")
    args = parser.parse_args(argv)
    report = inspect_sources()
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else source_diagnostic_text(report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
