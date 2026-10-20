"""Static source/AST/JSON/LF check; no dataset or scientific backend import."""
import ast
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))

from shared_weather_benchmark.provenance import HERE,verify_sources,atomic_json


def main():
    count = 0
    for p in HERE.rglob("*"):
        if not p.is_file() or "local_validation" in p.parts or "__pycache__" in p.parts:
            continue
        if p.suffix in {".py",".sh",".md",".json"}:
            if b"\r" in p.read_bytes():
                raise ValueError("CRLF/CR rejected: " + str(p))
            if p.suffix == ".py":
                ast.parse(p.read_text("utf-8"),filename=str(p))
            if p.suffix == ".json":
                json.loads(p.read_text("utf-8"))
            count += 1
    shell = (HERE/"run_server.sh").read_text("utf-8")
    if "pipefail" in shell or "[[" in shell or "\r" in shell:
        raise ValueError("Non-POSIX shell feature")
    result = dict(status="passed",files=count,source_identity=verify_sources(),
                  actual_shell_parser="not confirmed locally; launcher requires sh -n on Linux")
    atomic_json(HERE/"local_validation"/"STATIC_CHECK.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
