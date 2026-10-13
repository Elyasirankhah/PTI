#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "scripts"))

from _lib.outcomes import load_regression_pids  # noqa: E402
from _lib.prediction_eval import (  # noqa: E402
    ml_classifier_specs,
    nested_loocv_classifier,
    nested_loocv_classifier_block,
)
from config import (  # noqa: E402
    CODING_SHEET,
    PV_PREDICTIONS_JSON,
    OUTPUT_DIR,
    PARTICIPANT_OUTCOMES,
    UNITS_JSONL,
)
from integrated_model import _split_blocks  # noqa: E402
from pv_regression_utils import build_pv_table, load_pv_labels  # noqa: E402
from regression_utils import (  # noqa: E402
    load_pv_from_coding_sheet,
    load_pv_from_inline,
    load_pv_from_predictions,
    load_strict_paragraphs,
)
from feature_utils import (  # noqa: E402
    build_extended_pv_features,
    drop_constants,
    merge_tables,
)

OUT_TXT = OUTPUT_DIR / "classifier_sensitivity.txt"
OUT_JSON = OUTPUT_DIR / "classifier_sensitivity.json"
NMF_PARTICIPANT = OUTPUT_DIR / "derived_themes_participant.csv"


def _matrix(table: list[dict[str, Any]], cols: list[str], outcome: str) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([int(r[outcome]) for r in table])
    X = np.array([[float(r.get(c) or 0) for c in cols] for r in table])
    return X, y


def _run_spec(
    spec_id: str,
    spec_name: str,
    run_fn: Callable[[Any, bool], dict[str, Any]],
    models: list[tuple[str, Any, bool]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, factory, scale in models:
        try:
            res = run_fn(factory, scale)
            rows.append({
                "feature_setup": spec_id,
                "feature_setup_label": spec_name,
                "classifier": model_name,
                "loocv_auc": res.get("loocv_auc"),
                "loocv_accuracy": res.get("loocv_accuracy"),
                "example_fold_features": res.get("example_fold_features"),
                "error": None,
            })
        except Exception as exc:
            rows.append({
                "feature_setup": spec_id,
                "feature_setup_label": spec_name,
                "classifier": model_name,
                "loocv_auc": None,
                "loocv_accuracy": None,
                "error": str(exc),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=UNITS_JSONL)
    ap.add_argument("--predictions", type=Path, default=PV_PREDICTIONS_JSON)
    ap.add_argument("--coding-sheet", type=Path, default=CODING_SHEET)
    ap.add_argument("--nmf-patient", type=Path, default=NMF_PARTICIPANT)
    ap.add_argument("--outcomes", type=Path, default=PARTICIPANT_OUTCOMES)
    ap.add_argument("--outcome", default="depressed")
    ap.add_argument("--pv-top-k", type=int, default=5)
    ap.add_argument("--nmf-top-k", type=int, default=4)
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of classifiers (default: all five)",
    )
    args = ap.parse_args()

    allowed = load_regression_pids(args.outcomes)
    paragraphs = load_strict_paragraphs(args.input, allowed)
    if not paragraphs:
        raise SystemExit("No strict participant paragraphs found.")

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
        raise SystemExit(
            "No PV subcode columns — copy pv_predictions.json to outputs/."
        )

    # PV-only table (same as pv_nested_model)
    pv_labels2 = load_pv_labels(
        paragraphs, predictions=args.predictions,
        coding_sheet=args.coding_sheet, prefer_predictions=True,
    )
    pv_table, _, pv_sub_cols = build_pv_table(
        paragraphs, pv_labels2, args.outcomes, args.outcome,
    )
    y_pv = np.array([int(r[args.outcome]) for r in pv_table])
    X_pv_raw = np.array([[float(r.get(c) or 0) for c in pv_sub_cols] for r in pv_table])
    X_pv, pv_kept, _ = drop_constants(X_pv_raw, pv_sub_cols)

    # NMF + combined blocks
    Xa, Xp, y_block, ash_kept, pv_kept_block = _split_blocks(
        table, ash_cols, pv_cols, args.outcome,
    )

    all_models = ml_classifier_specs(len(y_pv))
    if args.models:
        want = set(args.models)
        models = [m for m in all_models if m[0] in want]
        if not models:
            raise SystemExit(f"No matching models in {args.models}")
    else:
        models = all_models

    results: list[dict[str, Any]] = []

    results.extend(_run_spec(
        "pv_nested5",
        f"PV nested top-{args.pv_top_k}",
        lambda factory, scale: nested_loocv_classifier(
            X_pv, y_pv, pv_kept, factory,
            needs_scaling=scale,
            select_top_k=min(args.pv_top_k, X_pv.shape[1]),
        ),
        models,
    ))

    results.extend(_run_spec(
        "nmf_nested4",
        f"NMF nested top-{args.nmf_top_k}",
        lambda factory, scale: nested_loocv_classifier(
            Xa, y_block, ash_kept, factory,
            needs_scaling=scale,
            select_top_k=min(args.nmf_top_k, Xa.shape[1]),
        ),
        models,
    ))

    results.extend(_run_spec(
        "combined_block",
        f"Combined nested {args.pv_top_k} PV + {args.nmf_top_k} NMF",
        lambda factory, scale: nested_loocv_classifier_block(
            Xa, Xp, ash_kept, pv_kept_block, y_block, factory,
            needs_scaling=scale,
            k_ash=args.nmf_top_k,
            k_pv=args.pv_top_k,
        ),
        models,
    ))

    n_pos = int(np.sum(y_pv == 1))
    lines = [
        "Demo three feature setups × ML classifiers (nested LOOCV)",
        "=" * 78,
        f"Predictions: {args.predictions}",
        f"Cohort: n={len(pv_table)} ({n_pos} depressed, {len(pv_table) - n_pos} non-depressed)",
        f"PV pool: {X_pv.shape[1]} subcodes  |  NMF pool: {Xa.shape[1]} themes",
        "Feature selection: inside each LOOCV training fold (unchanged from primary).",
        "",
        f"{'Feature setup':<42} {'Classifier':<16} {'AUC':>8} {'Acc':>8}",
        "-" * 78,
    ]
    for r in results:
        auc = r.get("loocv_auc")
        acc = r.get("loocv_accuracy")
        lines.append(
            f"{r['feature_setup_label'][:42]:<42} "
            f"{r['classifier']:<16} "
            f"{(auc if auc is not None else 'NA'):>8} "
            f"{(acc if acc is not None else 'NA'):>8}"
        )
        if r.get("error"):
            lines.append(f"  ERROR: {r['error']}")

    payload = {
        "predictions": str(args.predictions.resolve()),
        "outcome": args.outcome,
        "n_participants": len(pv_table),
        "pv_pool": int(X_pv.shape[1]),
        "nmf_pool": int(Xa.shape[1]),
        "pv_top_k": args.pv_top_k,
        "nmf_top_k": args.nmf_top_k,
        "classifiers": [m[0] for m in models],
        "results": results,
    }

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {OUT_TXT}")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
