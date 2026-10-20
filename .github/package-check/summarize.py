#!/usr/bin/env python3
"""Verify completed, identical-source matrix runs before reporting success."""
import argparse
import csv
from email.parser import Parser
import hashlib
import json
from pathlib import Path


def summarize(run, selected=None):
    manifest = (run / "source.sha256").read_text().split()
    archive = run / "source" / manifest[1]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != manifest[0]:
        raise ValueError(f"Source checksum mismatch: {run.name}")
    matrix = json.loads((run / "matrix.json").read_text())
    if selected is not None:
        matrix = [row for row in matrix if row["target"] == selected]
        if len(matrix) != 1:
            raise ValueError(f"Expected exactly one selected target: {selected}")
    result = []
    for target in matrix:
        if target["status"] != "PASS":
            raise ValueError(f"Unsuccessful target: {target['target']}")
        root = run / target["target"] / "results"
        with (root / "status.csv").open(newline="") as handle:
            stages = {r["stage"]: int(r["exit_code"]) for r in csv.DictReader(handle)}
        required = {"install", "smoke", "check", "check_complete",
                    "check_errors_warnings", "compiler_diagnostics", "check_notes_reviewed"}
        if not required <= stages.keys() or any(stages.values()):
            raise ValueError(f"Incomplete or failed stages: {target['target']}")
        logs = list(root.rglob("00check.log"))
        if len(logs) != 1:
            raise ValueError(f"Expected one final check log: {target['target']}")
        lines = logs[0].read_text(errors="replace").splitlines()
        status = [x for x in lines if x.startswith("Status:")]
        if "* DONE" not in lines or len(status) != 1 or any(
            word in status[0] for word in ("ERROR", "WARNING")
        ):
            raise ValueError(f"Final R check not clean: {target['target']}")
        package = archive.name.split("_", 1)[0]
        libraries = root / "library" / package / "libs"
        native = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in libraries.rglob("*") if p.suffix in (".so", ".dll", ".dylib")}
        description = root / "library" / package / "DESCRIPTION"
        metadata = Parser().parsestr(description.read_text())
        if metadata.get("NeedsCompilation", "no").strip() == "yes" and not native:
            raise ValueError(f"Installed native library missing: {target['target']}")
        result.append({**target, "source_sha256": digest,
                       "native_sha256": native, "R_check": status[0]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="*")
    parser.add_argument("--select", action="append", default=[],
                        metavar="TARGET=RUN",
                        help="Select an explicit successful profile after a retry; retain failed runs")
    parser.add_argument("--expect", required=True, help="Comma-separated expected target names")
    args = parser.parse_args()
    if not args.runs and not args.select:
        parser.error("Supply runs or explicit TARGET=RUN selections")
    rows = [row for run in args.runs for row in summarize(run)]
    for selection in args.select:
        name, separator, path = selection.partition("=")
        if not separator or not name or not path:
            parser.error("Selections must use TARGET=RUN")
        rows.extend(summarize(Path(path), selected=name))
    names = [row["target"] for row in rows]
    if len(names) != len(set(names)) or set(names) != set(args.expect.split(",")):
        raise ValueError("Missing, duplicate or unexpected profiles in the supplied runs")
    if len({row["source_sha256"] for row in rows}) != 1:
        raise ValueError("Runs tested different source archives")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
