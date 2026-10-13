#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "scripts"))

from _lib.outcomes import load_regression_pids  # noqa: E402
from _lib.prediction_eval import (  # noqa: E402
    nested_loocv_logistic,
    nested_loocv_logistic_block,
    permutation_pvalue,
    permutation_pvalue_nested_logistic_block,
    univariate_screen,
)
from config import (  # noqa: E402
    CODING_SHEET,
    PV_PREDICTIONS_JSON,
    OUTPUT_DIR,
    PARTICIPANT_OUTCOMES,
    UNITS_JSONL,
)
from regression_utils import (  # noqa: E402
    load_pv_from_coding_sheet,
    load_pv_from_inline,
    load_pv_from_predictions,
    load_strict_paragraphs,
    run_model,
    write_csv,
)
from feature_utils import (  # noqa: E402
    build_extended_pv_features,
    drop_constants,
    merge_tables,
)

OUT_TXT = OUTPUT_DIR / "integrated_model.txt"
OUT_JSON = OUTPUT_DIR / "integrated_model.json"
OUT_TABLE = OUTPUT_DIR / "integrated_patient_table.csv"
NMF_PARTICIPANT = OUTPUT_DIR / "derived_themes_participant.csv"


def _matrix(table: list[dict[str, Any]], cols: list[str], outcome: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    use = [c for c in cols if c in table[0]]
    y = np.array([int(r[outcome]) for r in table])
    X = np.array([[float(r.get(c) or 0) for c in use] for r in table])
    return X, y, use


def _split_blocks(
    table: list[dict[str, Any]],
    ash_cols: list[str],
    pv_cols: list[str],
    outcome: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    Xa, y, ash_kept = _matrix(table, ash_cols, outcome)
    Xp, _, pv_kept = _matrix(table, pv_cols, outcome)
    Xa, ash_kept, _ = drop_constants(Xa, ash_kept)
    Xp, pv_kept, _ = drop_constants(Xp, pv_kept)
    return Xa, Xp, y, ash_kept, pv_kept


def _eval_block_nested(
    table: list[dict[str, Any]],
    ash_cols: list[str],
    pv_cols: list[str],
    outcome: str,
    *,
    k_ash: int,
    k_pv: int,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    """Top NMF themes and top PV sub-codes per fold, concatenated into one logistic model."""
    Xa, Xp, y, ash_kept, pv_kept = _split_blocks(table, ash_cols, pv_cols, outcome)
    if Xa.shape[1] == 0 or Xp.shape[1] == 0:
        return {"name": f"Block {k_pv} PV + {k_ash} NMF", "error": "empty block"}

    nested = nested_loocv_logistic_block(
        Xa, Xp, ash_kept, pv_kept, y, k_ash=k_ash, k_pv=k_pv,
    )
    perm = permutation_pvalue_nested_logistic_block(
        Xa, Xp, ash_kept, pv_kept, y,
        k_ash=k_ash, k_pv=k_pv,
        observed_auc=nested.get("loocv_auc"),
        n_perm=n_perm,
        seed=seed,
    )
    n_feat = min(k_ash, Xa.shape[1]) + min(k_pv, Xp.shape[1])
    return {
        "name": f"Block: {k_pv} PV + {k_ash} NMF → one regression",
        "method": "block_nested_one_regression",
        "matrix": f"29 x {n_feat} ({k_pv} PV + {k_ash} NMF)",
        "k_pv": k_pv,
        "k_ash": k_ash,
        "auc": nested.get("loocv_auc"),
        "accuracy": nested.get("loocv_accuracy"),
        "perm_p": perm.get("permutation_p_auc"),
        "example_fold_features": nested.get("example_fold_features"),
        "note": "One model: Depression ~ PV features + NMF features; top-k per block inside each fold",
    }


def _eval_block_screen(
    table: list[dict[str, Any]],
    ash_cols: list[str],
    pv_cols: list[str],
    outcome: str,
    *,
    k_ash: int,
    k_pv: int,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    """Screen top-k per block on full n=29, concat columns, one LOOCV (exploratory/leaky)."""
    Xa, Xp, y, ash_kept, pv_kept = _split_blocks(table, ash_cols, pv_cols, outcome)
    if Xa.shape[1] == 0 or Xp.shape[1] == 0:
        return {"name": f"Block screen {k_pv} PV + {k_ash} NMF", "error": "empty block"}

    top_ash = univariate_screen(Xa, y, ash_kept, top_k=min(k_ash, Xa.shape[1]))
    top_pv = univariate_screen(Xp, y, pv_kept, top_k=min(k_pv, Xp.shape[1]))
    used = [t["feature"] for t in top_ash] + [t["feature"] for t in top_pv]

    res = run_model(
        f"Block screen: {k_pv} PV + {k_ash} NMF → one regression",
        table, used, outcome, n_perm=n_perm, seed=seed,
    )
    res["method"] = "block_screen_one_regression"
    res["matrix"] = f"29 x {len(used)} ({k_pv} PV + {k_ash} NMF)"
    res["k_pv"] = k_pv
    res["k_ash"] = k_ash
    res["nmf_features"] = [t["feature"] for t in top_ash]
    res["pv_features"] = [t["feature"] for t in top_pv]
    res["note"] = "Exploratory: feature pick on full sample before LOOCV (slight leakage)"
    return res


def _eval_topk_screen(
    table: list[dict[str, Any]],
    pool_cols: list[str],
    outcome: str,
    *,
    k: int,
    name: str,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    """Legacy: only k features total from the merged pool."""
    X_raw, y, cols = _matrix(table, pool_cols, outcome)
    X, kept, dropped = drop_constants(X_raw, cols)
    if X.shape[1] == 0:
        return {"name": name, "error": "no features", "k": k}

    top = univariate_screen(X, y, kept, top_k=min(k, X.shape[1]))
    used = [t["feature"] for t in top]
    idx = [kept.index(c) for c in used]
    X_k = X[:, idx]

    res = run_model(name, table, used, outcome, n_perm=n_perm, seed=seed)
    res["method"] = f"combined_top{k}_screen"
    res["matrix"] = f"29 x {len(used)}"
    res["pool_dropped_constant"] = dropped
    res["univariate_top"] = top
    return res


def _eval_nested_topk(
    table: list[dict[str, Any]],
    pool_cols: list[str],
    outcome: str,
    *,
    k: int,
    name: str,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    X_raw, y, cols = _matrix(table, pool_cols, outcome)
    X, kept, dropped = drop_constants(X_raw, cols)
    if X.shape[1] == 0:
        return {"name": name, "error": "no features", "k": k}

    top_k = min(k, X.shape[1])
    nested = nested_loocv_logistic(X, y, kept, select_top_k=top_k)
    perm = permutation_pvalue(
        X, y,
        observed_auc=nested.get("loocv_auc"),
        n_perm=n_perm,
        seed=seed + k,
    )
    return {
        "name": name,
        "method": f"combined_nested_top{k}",
        "matrix": f"29 x {top_k}",
        "k": k,
        "n_features_pool": X.shape[1],
        "dropped_constant": dropped,
        "auc": nested.get("loocv_auc"),
        "accuracy": nested.get("loocv_accuracy"),
        "perm_p": perm.get("permutation_p_auc"),
        "example_fold_features": nested.get("example_fold_features"),
        "note": "Legacy: k features total from the merged pool inside each fold",
    }


def _parse_block(s: str) -> tuple[int, int]:
    """Parse '5,4' -> (k_pv=5, k_ash=4)."""
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected k_pv,k_ash e.g. 5,4 got {s!r}")
    return parts[0], parts[1]


def _format_block(r: dict[str, Any]) -> list[str]:
    matrix = r.get("matrix")
    if matrix is None:
        matrix = f"29 x {r.get('n_features', '?')}"
    lines = [
        "-" * 70,
        f"MODEL: {r.get('name', '?')}",
        f"  Method: {r.get('method', r.get('name', ''))}",
        f"  Matrix: {matrix}",
    ]
    if r.get("nmf_features") or r.get("pv_features"):
        if r.get("nmf_features"):
            lines.append(f"  NMF cols: {', '.join(r['nmf_features'])}")
        if r.get("pv_features"):
            lines.append(f"  PV cols: {', '.join(r['pv_features'][:8])}")
            if len(r["pv_features"]) > 8:
                lines.append(f"    ... +{len(r['pv_features']) - 8} more PV")
    if r.get("features"):
        lines.append(f"  Features ({r.get('n_features', len(r['features']))}): {', '.join(r['features'][:12])}")
        if len(r["features"]) > 12:
            lines.append(f"    ... +{len(r['features']) - 12} more")
    if r.get("example_fold_features"):
        lines.append(f"  Example fold features: {', '.join(r['example_fold_features'])}")
    if r.get("auc") is not None:
        lines.append(f"  LOOCV AUC:      {r.get('auc')}")
        lines.append(f"  Permutation p:  {r.get('perm_p')}")
    elif r.get("error"):
        lines.append(f"  ERROR: {r['error']}")
    if r.get("note"):
        lines.append(f"  Note: {r['note']}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=UNITS_JSONL)
    ap.add_argument("--predictions", type=Path, default=PV_PREDICTIONS_JSON)
    ap.add_argument("--coding-sheet", type=Path, default=CODING_SHEET)
    ap.add_argument("--nmf-patient", type=Path, default=NMF_PARTICIPANT)
    ap.add_argument("--outcomes", type=Path, default=PARTICIPANT_OUTCOMES)
    ap.add_argument("--outcome", default="depressed")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--blocks",
        nargs="+",
        default=["5,4", "5,6"],
        help="Block specs as k_pv,k_nmf e.g. 5,4 → one regression",
    )
    ap.add_argument(
        "--legacy-pooled-k",
        type=int,
        nargs="*",
        default=[],
        help="Optional legacy: k features total from the merged pool",
    )
    ap.add_argument(
        "--only-concat",
        action="store_true",
        help="Run only the concatenated model (first --blocks spec)",
    )
    args = ap.parse_args()

    if args.only_concat:
        args.legacy_pooled_k = []
        # Single-block quick run: use only the first --blocks spec (respects e.g. 3,2).
        args.blocks = args.blocks[:1]

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
    write_csv(table, OUT_TABLE)

    ash_cols = [c for c in table[0] if c.endswith("_density")]
    pv_cols = [c for c in pv_pools.get("pv_subcode_raw", []) if c in table[0]]
    combined_pool = ash_cols + pv_cols

    if args.only_concat and not pv_cols:
        raise SystemExit(
            "No PV subcode columns found — cannot run concatenated model.\n"
            f"  Predictions: {args.predictions}\n"
            f"  Coding sheet: {args.coding_sheet}\n"
            "Copy pv_predictions.json from the compute environment to outputs/, then rerun."
        )

    results: list[dict[str, Any]] = []

    if not args.only_concat:
        results.append(run_model(
            "NMF-only (NMF theme densities)", table, ash_cols, args.outcome,
            n_perm=args.n_perm, seed=args.seed,
        ))
        results[-1]["method"] = "nmf_only"
        results[-1]["matrix"] = f"29 x {results[-1].get('n_features', '?')}"

        results.append(run_model(
            "PV-only (subcode densities)", table, pv_cols, args.outcome,
            n_perm=args.n_perm, seed=args.seed + 1,
        ))
        results[-1]["method"] = "pv_subcode_only"
        results[-1]["matrix"] = f"29 x {results[-1].get('n_features', '?')}"

        results.append(run_model(
            "Combined-all (NMF + PV concatenated)", table, combined_pool, args.outcome,
            n_perm=args.n_perm, seed=args.seed + 2,
        ))
        results[-1]["method"] = "combined_all_concat"
        results[-1]["matrix"] = f"29 x {results[-1].get('n_features', '?')}"
        results[-1]["note"] = "One regression on ALL NMF + ALL PV columns concatenated"

    for block in args.blocks:
        k_pv, k_ash = _parse_block(block)
        results.append(_eval_block_nested(
            table, ash_cols, pv_cols, args.outcome,
            k_ash=k_ash, k_pv=k_pv,
            n_perm=args.n_perm, seed=args.seed + 100 + k_pv * 10 + k_ash,
        ))
        if not args.only_concat:
            results.append(_eval_block_screen(
                table, ash_cols, pv_cols, args.outcome,
                k_ash=k_ash, k_pv=k_pv,
                n_perm=args.n_perm, seed=args.seed + 200 + k_pv * 10 + k_ash,
            ))

    if not args.only_concat:
        for k in sorted(set(args.legacy_pooled_k)):
            results.append(_eval_topk_screen(
                table, combined_pool, args.outcome,
                k=k,
                name=f"[legacy] {k} features total from merged pool (screen)",
                n_perm=args.n_perm,
                seed=args.seed + 10 + k,
            ))
            results.append(_eval_nested_topk(
                table, combined_pool, args.outcome,
                k=k,
                name=f"[legacy] {k} features total from merged pool (nested)",
                n_perm=args.n_perm,
                seed=args.seed + 20 + k,
            ))

    n_pos = sum(r[args.outcome] for r in table)
    primary = results[0] if args.only_concat and results else None

    if args.only_concat and primary:
        k_pv, k_ash = _parse_block(args.blocks[0])
        lines = [
            f"Demo concatenated model — ONE AUC ({k_pv} nested PV + {k_ash} NMF themes)",
            "=" * 72,
            f"Predictions: {args.predictions}",
            f"Cohort: n={len(table)} ({n_pos} depressed, {len(table) - n_pos} non-depressed)",
            f"Design: Depression ~ top-{k_pv} PV subcodes + top-{k_ash} NMF themes (one logistic)",
            f"Matrix: {primary.get('matrix', f'29 x {k_pv + k_ash}')}",
            "",
            f"LOOCV AUC:       {primary.get('auc')}",
            f"Permutation p:   {primary.get('perm_p')}",
        ]
        if primary.get("example_fold_features"):
            lines.append(f"Example fold:    {', '.join(primary['example_fold_features'])}")
    else:
        lines = [
            "Combined NMF + PV — one regression",
            "=" * 72,
            f"Predictions: {args.predictions}",
            f"Cohort: n={len(table)} ({n_pos} depressed, {len(table) - n_pos} non-depressed)",
            f"Paragraphs: {len(paragraphs)}",
            f"NMF theme columns: {len(ash_cols)}  |  PV subcode columns: {len(pv_cols)}",
            "",
            "Stack PV and NMF theme columns side-by-side in one logistic model",
            "  Depression ~ PV_1 + ... + PV_k + NMF_1 + ... + NMF_m",
            f"  Block specs: {args.blocks}  (k_pv,k_ash → 29×[k_pv+k_ash])",
            "",
            f"{'Model':<52} {'Matrix':<10} {'AUC':>8} {'Perm p':>8}",
            "-" * 72,
        ]
        for r in results:
            if r.get("auc") is None:
                continue
            lines.append(
                f"{r.get('name', '')[:52]:<52} "
                f"{str(r.get('matrix', '')):<10} "
                f"{r.get('auc'):>8} "
                f"{r.get('perm_p'):>8}"
            )
        lines.append("")
        for r in results:
            lines.extend(_format_block(r))

    payload = {
        "predictions": str(args.predictions.resolve()),
        "outcome": args.outcome,
        "n_participants": len(table),
        "nmf_cols": len(ash_cols),
        "pv_subcode_cols": len(pv_cols),
        "combined_pool_size": len(combined_pool),
        "results": results,
    }

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {OUT_TXT}")
    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {OUT_TABLE}")


if __name__ == "__main__":
    main()
