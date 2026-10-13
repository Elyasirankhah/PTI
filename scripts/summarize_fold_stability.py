#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "scripts"))

from _lib.outcomes import load_regression_pids  # noqa: E402
from _lib.prediction_eval import _univariate_rank_train  # noqa: E402
from config import CODING_SHEET, PV_PREDICTIONS_JSON, OUTPUT_DIR, PARTICIPANT_OUTCOMES, UNITS_JSONL  # noqa: E402
from integrated_model import _split_blocks  # noqa: E402
from regression_utils import (  # noqa: E402
    load_pv_from_coding_sheet,
    load_pv_from_inline,
    load_pv_from_predictions,
    load_strict_paragraphs,
)
from feature_utils import build_extended_pv_features, merge_tables  # noqa: E402

NMF_PARTICIPANT = OUTPUT_DIR / "derived_themes_participant.csv"
OUT_TXT = OUTPUT_DIR / "pv_fold_stability.txt"
OUT_JSON = OUTPUT_DIR / "pv_fold_stability.json"


def collect_pv_fold_selections(
    table: list[dict],
    ash_cols: list[str],
    pv_cols: list[str],
    outcome: str,
    *,
    k_pv: int,
    k_ash: int,
) -> list[list[str]]:
    from sklearn.model_selection import LeaveOneOut

    Xa, Xp, y, ash_kept, pv_kept = _split_blocks(table, ash_cols, pv_cols, outcome)
    loo = LeaveOneOut()
    fold_pv: list[list[str]] = []

    for train_idx, _test_idx in loo.split(y):
        Xa_train = Xa[train_idx]
        Xp_train = Xp[train_idx]
        y_train = y[train_idx]

        ash_idx = _univariate_rank_train(
            Xa_train, y_train, ash_kept, top_k=min(k_ash, Xa.shape[1])
        )
        pv_idx = _univariate_rank_train(
            Xp_train, y_train, pv_kept, top_k=min(k_pv, Xp.shape[1])
        )
        if not ash_idx:
            ash_idx = list(range(min(k_ash, Xa.shape[1])))
        if not pv_idx:
            pv_idx = list(range(min(k_pv, Xp.shape[1])))

        fold_pv.append([pv_kept[j] for j in pv_idx])

    return fold_pv


def summarize(fold_pv: list[list[str]], *, pct_threshold: float = 0.80) -> dict:
    n_folds = len(fold_pv)
    counts: Counter[str] = Counter()
    for picks in fold_pv:
        counts.update(picks)

    min_folds = math.ceil(pct_threshold * n_folds)
    stable = sorted(f for f, c in counts.items() if c >= min_folds)
    union_unique = sorted(counts.keys())

    per_fold_sizes = [len(set(p)) for p in fold_pv]
    return {
        "n_folds": n_folds,
        "k_pv_per_fold": len(fold_pv[0]) if fold_pv else 0,
        "union_unique_subcodes": len(union_unique),
        "mean_unique_per_fold": float(np.mean(per_fold_sizes)) if per_fold_sizes else 0.0,
        "pct_threshold": pct_threshold,
        "min_folds_for_threshold": min_folds,
        "n_stable_ge_threshold": len(stable),
        "stable_subcodes": stable,
        "selection_counts": dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=UNITS_JSONL)
    ap.add_argument("--predictions", type=Path, default=PV_PREDICTIONS_JSON)
    ap.add_argument("--coding-sheet", type=Path, default=CODING_SHEET)
    ap.add_argument("--nmf-patient", type=Path, default=NMF_PARTICIPANT)
    ap.add_argument("--outcomes", type=Path, default=PARTICIPANT_OUTCOMES)
    ap.add_argument("--outcome", default="depressed")
    ap.add_argument("--k-pv", type=int, default=5)
    ap.add_argument("--k-ash", type=int, default=4)
    ap.add_argument("--pct", type=float, default=0.80)
    args = ap.parse_args()

    pids = load_regression_pids(args.outcomes)
    paragraphs = load_strict_paragraphs(args.input, pids)
    rids = {p["record_id"] for p in paragraphs}
    rows_by_id = {p["record_id"]: p for p in paragraphs}

    if args.predictions.is_file():
        pv_labels = load_pv_from_predictions(args.predictions, rids, rows_by_id)
    else:
        pv_labels = load_pv_from_coding_sheet(args.coding_sheet, rids)
        if not pv_labels:
            pv_labels = load_pv_from_inline(paragraphs, rids)

    pv_densities, pv_pools = build_extended_pv_features(paragraphs, pv_labels)
    table = merge_tables(args.nmf_patient, pv_densities, pv_pools, args.outcomes, args.outcome)

    ash_cols = [c for c in table[0] if c.endswith("_density")]
    pv_cols = [c for c in pv_pools.get("pv_subcode_raw", []) if c in table[0]]
    if not pv_cols:
        raise SystemExit("No PV subcode columns — need pv_predictions.json on compute environment.")

    fold_pv = collect_pv_fold_selections(
        table, ash_cols, pv_cols, args.outcome, k_pv=args.k_pv, k_ash=args.k_ash
    )
    stats = summarize(fold_pv, pct_threshold=args.pct)

    lines = [
        "PV nested fold-selection stability (integrated block design)",
        f"Design: top-{args.k_pv} PV + top-{args.k_ash} NMF per LOOCV fold",
        f"Folds: {stats['n_folds']}",
        f"Union of unique PV subcodes selected across folds: {stats['union_unique_subcodes']}",
        f"Mean unique subcodes per fold: {stats['mean_unique_per_fold']:.1f}",
        f"Selected in >={int(args.pct * 100)}% of folds (>={stats['min_folds_for_threshold']}/{stats['n_folds']}): "
        f"{stats['n_stable_ge_threshold']}",
        "",
        "Stable subcodes:",
    ]
    for f in stats["stable_subcodes"]:
        lines.append(f"  {f}  (n={stats['selection_counts'][f]}/{stats['n_folds']})")
    lines.extend(["", "All selection counts (descending):"])
    for f, c in stats["selection_counts"].items():
        lines.append(f"  {c:2d}/{stats['n_folds']}  {f}")

    text = "\n".join(lines)
    print(text)

    payload = {
        "k_pv": args.k_pv,
        "k_ash": args.k_ash,
        **stats,
    }
    OUT_TXT.write_text(text + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_TXT}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
