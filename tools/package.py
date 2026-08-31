"""Build the submission archive for the Devpost file upload.

Deliberately not "everything in the repo". The submission rules name what must
be included -- the agent entry file, its helper modules, setup instructions, a
report covering method and limitations, and a disclosure of latency, token usage
and cost -- and separately name what must not be. This selects for that.

What is left out, and why:

* the 60 MB catalog and its 19 MB archive -- the organizer's own frozen data,
  which they already have, and which alone exceeds the 35 MB upload limit;
* the organizer's own documents under `docs/` (specification, submission rules,
  API contract, evaluation config) -- theirs, not ours, and the rules say not to
  ship copied organizer files;
* the demo-video script and narration -- production notes for the recording, of
  no use to a judge;
* scratch measurement output at the repo root -- the curated copies live in
  `results/` and are what the README's tables cite;
* `.git/`, `__pycache__/`, and the generated held-out session dump.

What is deliberately kept, against a first instinct to strip it: `tools/`. Those
are not incidental scripts. The README's central claims -- held-out
generalization, the component ablation, the paraphrase stress test, the cost and
memory profile -- are each produced by one of them, and `results/` holds their
raw output. Removing them would leave the numbers unverifiable, which is the
opposite of what the report is for.

    python tools/package.py
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files and directories that go in, in the order a reader should meet them.
INCLUDE_FILES = [
    "README.md",              # the report: method, architecture, limitations
    "RUN.md",                 # setup and reproduction, one page
    "SUBMISSION_CHECKLIST.md",  # maps these files to the official deliverables
    "requirements.txt",
    "DATA_ATTRIBUTION.md",
    "SHA256SUMS",
]

INCLUDE_DIRS = [
    "starter",    # the required Agent entry point
    "copilot",    # helper modules -- the actual system
    "tools",      # the harnesses behind every number in the report
    "tests",
    "results",    # raw evidence for the report's tables
    "evaluator",  # the organizer's scorer, unmodified, so the zip runs as-is
]

# Never shipped, wherever they appear.
SKIP_NAMES = {"__pycache__", ".git", ".pytest_cache", ".DS_Store"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SKIP_SPECIFIC = {
    "tools/record.py",          # a screen-recording aid for the demo video
    "tools/package.py",         # this script
}


def wanted(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    if relative in SKIP_SPECIFIC:
        return False
    if path.suffix in SKIP_SUFFIXES:
        return False
    return not any(part in SKIP_NAMES for part in path.parts)


def collect() -> list[Path]:
    chosen: list[Path] = []
    for name in INCLUDE_FILES:
        path = ROOT / name
        if path.exists():
            chosen.append(path)
    for name in INCLUDE_DIRS:
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and wanted(path):
                chosen.append(path)
    # The public session file is small, is not private evaluation data, and the
    # reproduction steps need it. The catalog is not: it is 60 MB.
    public_set = ROOT / "data" / "public_set.jsonl"
    if public_set.exists():
        chosen.append(public_set)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the submission archive")
    parser.add_argument("--output", default=str(ROOT / "shopping-copilot-submission.zip"))
    args = parser.parse_args()

    files = collect()
    target = Path(args.output)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path("shopping-copilot") / path.relative_to(ROOT))

    size = target.stat().st_size
    print(f"{target}")
    print(f"  {len(files)} files, {size / 1024 / 1024:.2f} MB "
          f"({100 * size / (35 * 1024 * 1024):.1f}% of the 35 MB limit)")
    top: dict[str, int] = {}
    for path in files:
        key = path.relative_to(ROOT).parts[0]
        top[key] = top.get(key, 0) + 1
    for key, count in sorted(top.items(), key=lambda kv: -kv[1]):
        print(f"    {key:14} {count:3d}")


if __name__ == "__main__":
    main()
