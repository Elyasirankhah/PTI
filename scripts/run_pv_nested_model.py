#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "scripts"))

from _lib.outcomes import load_regression_pids  # noqa: E402
from _lib.prediction_eval import (  # noqa: E402
    nested_loocv_logistic,
    permutation_pvalue_nested_logistic,
)
from config import CODING_SHEET, PV_PREDICTIONS_JSON, OUTPUT_DIR, PARTICIPANT_OUTCOMES, UNITS_JSONL  # noqa: E402
from pv_regression_utils import build_pv_table, load_pv_labels  # noqa: E402
from regression_utils import load_strict_paragraphs  # noqa: E402
from feature_utils import drop_constants  # noqa: E402

def _out_paths(tag: str | None) -> tuple[Path, Path]:
    if tag:
        safe = tag.replace("/", "_").strip()
        return (
            OUTPUT_DIR / f"pv_nested_model_{safe}.txt",
            OUTPUT_DIR / f"pv_nested_model_{safe}.json",
        )
    return (
        OUTPUT_DIR / "pv_nested_model.txt",
        OUTPUT_DIR / "pv_nested_model.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Nested top-5 PV permutation test")
    ap.add_argument("--input", type=Path, default=UNITS_JSONL)
    ap.add_argument("--predictions", type=Path, default=PV_PREDICTIONS_JSON)
    ap.add_argument("--coding-sheet", type=Path, default=CODING_SHEET)
    ap.add_argument("--outcomes", type=Path, default=PARTICIPANT_OUTCOMES)
    ap.add_argument("--outcome", default="depressed")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--tag",
        default=None,
        help="Output suffix, e.g. 8b -> pv_nested_model_8b.txt (for multi-model runs)",
    )
    args = ap.parse_args()
    out_txt, out_json = _out_paths(args.tag)

    allowed = load_regression_pids(args.outcomes)
    paragraphs = load_strict_paragraphs(args.input, allowed)
    if not paragraphs:
        raise SystemExit("No strict participant paragraphs found.")

    pv_labels = load_pv_labels(
        paragraphs,
        predictions=args.predictions,
        coding_sheet=args.coding_sheet,
        prefer_predictions=True,
    )
    n_labeled = sum(1 for p in paragraphs if pv_labels.get(p["record_id"]))
    if n_labeled == 0:
        raise SystemExit(
            "No PV labels found. Copy pv_predictions.json to outputs/ "
            f"or refresh {args.coding_sheet}."
        )

    table, _code_cols, sub_cols = build_pv_table(
        paragraphs, pv_labels, args.outcomes, args.outcome,
    )
    if not table:
        raise SystemExit("No participants with outcomes and PV subcode features.")

    y = np.array([int(r[args.outcome]) for r in table])
    X_raw = np.array([[float(r.get(c) or 0) for c in sub_cols] for r in table])
    X, kept, dropped = drop_constants(X_raw, sub_cols)
    if X.shape[1] == 0:
        raise SystemExit("All PV subcode features constant.")

    nested = nested_loocv_logistic(
        X, y, kept, select_top_k=min(args.top_k, X.shape[1])
    )
    observed_auc = nested.get("loocv_auc")
    perm = permutation_pvalue_nested_logistic(
        X, y, kept,
        select_top_k=min(args.top_k, X.shape[1]),
        observed_auc=observed_auc,
        n_perm=args.n_perm,
        seed=args.seed,
    )

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    pred_note = f"Predictions: {args.predictions.resolve()}"
    lines = [
        "Demo PV nested top-k permutation test",
        "=" * 60,
        pred_note,
        f"Outcome: {args.outcome}  |  n={len(table)} ({n_pos} depressed, {n_neg} non-depressed)",
        f"Paragraphs: {len(paragraphs)}  with >=1 PV label: {n_labeled}",
        f"Subcode pool: {X.shape[1]} features (dropped constant: {len(dropped)})",
        f"Method: nested LOOCV top-{args.top_k} + logistic regression",
        f"Permutations: {args.n_perm}  seed={args.seed}",
        "",
        f"LOOCV AUC:        {observed_auc}",
        f"Permutation p:    {perm.get('permutation_p_auc')}",
        f"Null AUC mean:    {perm.get('null_auc_mean')}",
        "",
        "Example fold features (first fold only):",
        f"  {', '.join(nested.get('example_fold_features') or [])}",
    ]
    payload = {
        "predictions": str(args.predictions.resolve()),
        "tag": args.tag,
        "outcome": args.outcome,
        "n_participants": len(table),
        "n_depressed": n_pos,
        "n_subcode_pool": int(X.shape[1]),
        "top_k": args.top_k,
        "loocv_auc": observed_auc,
        "perm_p": perm.get("permutation_p_auc"),
        "null_auc_mean": perm.get("null_auc_mean"),
        "n_perm": args.n_perm,
        "seed": args.seed,
        "dropped_constant": dropped,
        "example_fold_features": nested.get("example_fold_features"),
    }

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {out_txt}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
