#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
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
    bootstrap_ci,
    loocv_logistic,
    permutation_pvalue,
    univariate_screen,
)
from _lib.sites import DEMO_SUBCORPUS  # noqa: E402
from config import CODING_SHEET, OUTPUT_DIR, PARTICIPANT_OUTCOMES, UNITS_JSONL  # noqa: E402
from regression_utils import (  # noqa: E402
    build_pv_person_densities,
    load_pv_from_coding_sheet,
    load_pv_from_inline,
    load_pv_from_predictions,
    load_strict_paragraphs,
    write_csv,
)
from _lib.participants import participant_id_from_source  # noqa: E402
from feature_utils import build_extended_pv_features  # noqa: E402

PREDICTIONS_DEFAULT = OUTPUT_DIR / "pv_predictions.json"
OUT_CSV = OUTPUT_DIR / "pv_regression_utils_participant.csv"
OUT_TXT = OUTPUT_DIR / "pv_regression_utils.txt"
OUT_JSON = OUTPUT_DIR / "pv_regression_utils.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_pv_labels(
    paragraphs: list[dict[str, Any]],
    *,
    predictions: Path | None,
    coding_sheet: Path,
    prefer_predictions: bool,
) -> dict[str, list[tuple[str, str]]]:
    valid_rids = {p["record_id"] for p in paragraphs}
    rows_by_id = {p["record_id"]: p for p in paragraphs}

    labels: dict[str, list[tuple[str, str]]] = {}
    if prefer_predictions and predictions and predictions.is_file():
        labels = dict(load_pv_from_predictions(predictions, valid_rids, rows_by_id))
    if not labels and coding_sheet.is_file():
        labels = dict(load_pv_from_coding_sheet(coding_sheet, valid_rids))
    if not labels:
        labels = dict(load_pv_from_inline(paragraphs, valid_rids))
    return labels


def build_pv_table(
    paragraphs: list[dict[str, Any]],
    pv_labels: dict[str, list[tuple[str, str]]],
    outcomes_path: Path,
    outcome_col: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    code_densities, code_cols = build_pv_person_densities(paragraphs, pv_labels)
    ext_densities, pools = build_extended_pv_features(paragraphs, pv_labels)
    sub_cols = pools["pv_subcode_raw"]

    outcomes = {
        r["participant_id"].strip().upper(): r
        for r in read_csv(outcomes_path)
        if r.get("subcorpus") == DEMO_SUBCORPUS
    }

    pids = sorted(set(code_densities) & set(outcomes))
    table: list[dict[str, Any]] = []
    for pid in pids:
        y_val = (outcomes[pid].get(outcome_col) or "").strip()
        if y_val not in {"0", "1"}:
            continue
        n_para = sum(
            1 for p in paragraphs
            if participant_id_from_source(p.get("source") or "").upper() == pid
        )
        row: dict[str, Any] = {
            "participant_id": pid,
            outcome_col: int(y_val),
            "n_paragraphs": n_para,
        }
        for col in code_cols:
            row[col] = code_densities.get(pid, {}).get(col, 0.0)
        ext = ext_densities.get(pid, {})
        for col in sub_cols:
            row[col] = ext.get(col, 0.0)
        table.append(row)

    return table, code_cols, sub_cols


def run_model(
    name: str,
    table: list[dict[str, Any]],
    feat_cols: list[str],
    outcome_col: str,
    *,
    n_perm: int = 2000,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    y = np.array([int(r[outcome_col]) for r in table])
    X = np.array([[float(r.get(c) or 0) for c in feat_cols] for r in table])

    keep = [j for j in range(X.shape[1]) if not np.all(X[:, j] == X[0, j])]
    dropped = [feat_cols[j] for j in range(X.shape[1]) if j not in keep]
    kept = [feat_cols[j] for j in keep]
    X = X[:, keep]

    if X.shape[1] == 0:
        return {
            "name": name,
            "n_features_input": len(feat_cols),
            "n_features_used": 0,
            "features_used": [],
            "dropped_constant": dropped,
            "auc": None,
            "note": "all constant",
        }

    res = loocv_logistic(X, y, C=1.0)
    auc = res["loocv_auc"]
    perm = permutation_pvalue(X, y, observed_auc=auc, n_perm=n_perm, seed=seed)
    ci = bootstrap_ci(y, res["predictions"], res["probabilities"], n_boot=n_boot, seed=seed)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    preds = np.array(res["predictions"])
    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))

    return {
        "name": name,
        "n_features_input": len(feat_cols),
        "n_features_used": len(kept),
        "features_used": kept,
        "dropped_constant": dropped,
        "auc": round(auc, 4),
        "accuracy": round(res["loocv_accuracy"], 4),
        "sensitivity": round(tp / n_pos, 4) if n_pos else 0,
        "specificity": round(tn / n_neg, 4) if n_neg else 0,
        "perm_p": round(perm.get("permutation_p_auc", 1.0), 4),
        "auc_ci95": [round(ci["auc_ci95"][0], 4), round(ci["auc_ci95"][1], 4)],
        "acc_ci95": [round(ci["accuracy_ci95"][0], 4), round(ci["accuracy_ci95"][1], 4)],
    }


def format_block(title: str, r: dict[str, Any]) -> str:
    if r.get("auc") is None:
        return f"{title}\n  {r.get('note', 'no result')}\n"
    lines = [
        title,
        f"  Features: {r['n_features_used']} / {r['n_features_input']}",
        f"  Used: {', '.join(r['features_used'][:12])}"
        + (" ..." if len(r["features_used"]) > 12 else ""),
    ]
    if r.get("dropped_constant"):
        lines.append(f"  Dropped (constant): {', '.join(r['dropped_constant'])}")
    lines += [
        f"  LOOCV AUC:         {r['auc']}",
        f"  LOOCV Accuracy:    {r['accuracy']}",
        f"  Sensitivity:       {r['sensitivity']}",
        f"  Specificity:       {r['specificity']}",
        f"  Permutation p:     {r['perm_p']}",
        f"  Bootstrap AUC CI:  {r['auc_ci95']}",
        f"  Bootstrap Acc CI:  {r['acc_ci95']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="PV-only LOOCV regression")
    ap.add_argument("--input", type=Path, default=UNITS_JSONL)
    ap.add_argument("--predictions", type=Path, default=PREDICTIONS_DEFAULT)
    ap.add_argument("--coding-sheet", type=Path, default=CODING_SHEET)
    ap.add_argument("--outcomes", type=Path, default=PARTICIPANT_OUTCOMES)
    ap.add_argument("--outcome", default="depressed")
    ap.add_argument("--prefer-predictions", action="store_true", default=True)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    allowed = load_regression_pids(args.outcomes)
    paragraphs = load_strict_paragraphs(args.input, allowed)
    if not paragraphs:
        raise SystemExit("No strict participant paragraphs found.")

    pv_labels = load_pv_labels(
        paragraphs,
        predictions=args.predictions,
        coding_sheet=args.coding_sheet,
        prefer_predictions=args.prefer_predictions,
    )
    n_labeled = sum(1 for p in paragraphs if pv_labels.get(p["record_id"]))
    print(f"PV-only regression (outcome={args.outcome})")
    print(f"  paragraphs={len(paragraphs)}  with_pv_labels={n_labeled}")

    table, code_cols, sub_cols = build_pv_table(
        paragraphs, pv_labels, args.outcomes, args.outcome,
    )
    if not table:
        raise SystemExit("No patients with outcomes and PV features.")

    n_pos = sum(r[args.outcome] for r in table)
    n_neg = len(table) - n_pos
    write_csv(table, OUT_CSV)

    r_codes = run_model(
        "PV codes (top-level TAB-PO)",
        table, code_cols, args.outcome,
        n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
    )
    r_subs = run_model(
        "PV subcodes (all raw code + sub-code pairs)",
        table, sub_cols, args.outcome,
        n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
    )

    y = np.array([int(r[args.outcome]) for r in table])
    X_sub = np.array([[float(r.get(c) or 0) for c in sub_cols] for r in table])
    keep = [j for j in range(X_sub.shape[1]) if not np.all(X_sub[:, j] == X_sub[0, j])]
    kept_sub = [sub_cols[j] for j in keep]
    X_sub = X_sub[:, keep]
    top5_screen = univariate_screen(X_sub, y, kept_sub, top_k=min(5, X_sub.shape[1]))
    top5_cols = [t["feature"] for t in top5_screen]
    r_top5 = run_model(
        "PV subcodes (top-5 univariate screen)",
        table, top5_cols, args.outcome,
        n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
    )

    print()
    print(format_block("MODEL 1: PV codes", r_codes))
    print(format_block("MODEL 2: PV subcodes (all)", r_subs))
    print(format_block("MODEL 3: PV subcodes (top-5 screen)", r_top5))
    print("Top-5 univariate LOOCV AUC (each alone):")
    for row in top5_screen:
        auc_val = row.get("loocv_auc")
        print(f"  {row['feature']}: {auc_val:.4f}" if auc_val is not None else f"  {row['feature']}: —")

    lines = [
        f"Demo PV-only regression (outcome={args.outcome})",
        "=" * 70,
        f"n={len(table)} ({n_pos} depressed, {n_neg} non-depressed)",
        f"Paragraphs: {len(paragraphs)}  with >=1 PV label: {n_labeled}",
        "Method: LOOCV logistic regression; native PV features (no NMF, no NMF mapping)",
        "Filter: only truly constant columns dropped",
        f"Participant table: {OUT_CSV.name}",
        "",
        "Compare to NMF (same cohort):",
        "  NMF 3 codes: AUC 0.58  |  NMF 6 themes: AUC 0.62  |  NMF 6 means: AUC 0.68",
        "",
        "-" * 70,
        format_block("MODEL 1: PV codes (top-level TAB-PO)", r_codes),
        "-" * 70,
        format_block("MODEL 2: PV subcodes (all raw pairs)", r_subs),
        "-" * 70,
        format_block("MODEL 3: PV subcodes (top-5 univariate screen)", r_top5),
        "-" * 70,
        "Univariate LOOCV AUC (each subcode alone):",
    ]
    for row in top5_screen:
        auc_val = row.get("loocv_auc")
        lines.append(
            f"  {row['feature']}: {round(float(auc_val), 4)}"
            if auc_val is not None
            else f"  {row['feature']}: —"
        )
    lines.extend([
        f"Combined (5 features): {r_top5.get('auc')}",
        "-" * 70,
    ])
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "n_participants": len(table),
                "n_paragraphs": len(paragraphs),
                "n_paragraphs_with_pv": n_labeled,
                "model_pv_codes": r_codes,
                "model_pv_subcodes_all": r_subs,
                "model_pv_subcodes_top5": r_top5,
                "univariate_top5": [
                    {
                        "feature": t["feature"],
                        "loocv_auc": round(float(t["loocv_auc"]), 4)
                        if t.get("loocv_auc") is not None
                        else None,
                    }
                    for t in top5_screen
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_TXT}")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
