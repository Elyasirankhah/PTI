#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "scripts"))

from _lib.io_utils import write_jsonl  # noqa: E402
from _lib.outcomes import load_regression_pids  # noqa: E402
from config import COHORT_UNITS_JSONL, PARTICIPANT_OUTCOMES, UNITS_JSONL  # noqa: E402
from regression_utils import load_strict_paragraphs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=UNITS_JSONL)
    ap.add_argument("--outcomes", type=Path, default=PARTICIPANT_OUTCOMES)
    ap.add_argument("--output", type=Path, default=COHORT_UNITS_JSONL)
    args = ap.parse_args()

    pids = load_regression_pids(args.outcomes)
    rows = load_strict_paragraphs(args.input, pids)
    if not rows:
        raise SystemExit("No strict participant paragraphs found for n=29 cohort.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} paragraphs -> {args.output}")
    print(f"  participants={len(pids)}  expected ~1203 paragraphs")


if __name__ == "__main__":
    main()
