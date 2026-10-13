#!/usr/bin/env python3
"""Run reviewer-requested validation analyses for the PTI demonstration.

1. Contraction-normalized NMF (k=6) vs default preprocessing — T4 robustness
2. Nested LOOCV AUCs: PV-only, NMF-only, integrated (5+4 block)
3. Paired permutation test: ΔAUC(integrated − PV) with label-shuffled null
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "scripts"))

from _lib.prediction_eval import (  # noqa: E402
    nested_loocv_logistic,
    nested_loocv_logistic_block,
    permutation_pvalue_nested_integrated_vs_pv,
    permutation_pvalue_nested_logistic,
    permutation_pvalue_nested_logistic_block,
)
from config import OUTPUT_DIR, PARTICIPANT_OUTCOMES, UNITS_JSONL  # noqa: E402
from feature_utils import drop_constants  # noqa: E402

OUT_TXT = OUTPUT_DIR / "pti_validation.txt"
OUT_JSON = OUTPUT_DIR / "pti_validation.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _matrix(
    rows: list[dict[str, str]],
    cols: list[str],
    outcome: str = "depressed",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    use = [c for c in cols if c in rows[0]]
    y = np.array([int(float(r[outcome])) for r in rows])
    X = np.array([[float(r.get(c) or 0) for c in use] for r in rows])
    return X, y, use


def _theme_rows_from_summary(summary: dict[str, Any]) -> list[dict[str, str]]:
    lines = []
    for t in summary.get("themes", []):
        top = t.get("top_word_summary", "")
        words = [w.strip() for w in top.split(",")[:8]]
        artifact_hits = [w for w in words if w.lower() in {"don", "didn", "dont", "didnt"}]
        lines.append({
            "theme": t.get("theme_label"),
            "name": t.get("theme_name") or "",
            "top_words": top,
            "artifact_tokens": ", ".join(artifact_hits) if artifact_hits else "",
            "n_primary_paragraphs": t.get("n_primary_paragraphs"),
        })
    return lines


def run_nmf_variant(
    *,
    k: int,
    preprocess: str,
    output_tag: str,
    seed: int,
    input_jsonl: Path,
    subcorpus: str | None,
    outcomes: Path,
) -> dict[str, Any]:
    tag = output_tag or preprocess
    cmd = [
        sys.executable,
        str(PIPELINE / "scripts" / "derive_nmf_themes.py"),
        "--input", str(input_jsonl),
        "--outcomes", str(outcomes),
        "--k", str(k),
        "--seed", str(seed),
        "--preprocess", preprocess,
        "--output-tag", tag,
    ]
    if subcorpus:
        cmd.extend(["--subcorpus", subcorpus])
    print(f"\n>>> NMF: k={k}, preprocess={preprocess}")
    subprocess.run(cmd, check=True, cwd=str(PIPELINE))
    summary_path = OUTPUT_DIR / f"derived_themes_{tag}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "preprocess": preprocess,
        "output_tag": tag,
        "summary_path": str(summary_path),
        "participant_csv": str(OUTPUT_DIR / f"derived_themes_{tag}_participant.csv"),
        "themes": _theme_rows_from_summary(summary),
        "k": summary.get("k"),
        "mean_coherence": summary.get("mean_coherence"),
    }


def load_merged_table(
    patient_table: Path,
    nmf_participant_csv: Path,
    outcome: str = "depressed",
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    """Merge PV subcode columns from saved patient table with NMF densities."""
    base = read_csv(patient_table)
    nmf_rows = read_csv(nmf_participant_csv)
    nmf_by_pid = {r["participant_id"]: r for r in nmf_rows}

    nmf_cols = [c for c in nmf_rows[0] if c.endswith("_density")]
    pv_cols = [c for c in base[0] if c.startswith("pvsub_")]

    merged: list[dict[str, str]] = []
    for row in base:
        pid = row["participant_id"]
        if pid not in nmf_by_pid:
            continue
        out = dict(row)
        for c in nmf_cols:
            out[c] = nmf_by_pid[pid].get(c, "0")
        merged.append(out)

    return merged, nmf_cols, pv_cols


def evaluate_models(
    table: list[dict[str, str]],
    nmf_cols: list[str],
    pv_cols: list[str],
    *,
    k_pv: int,
    k_ash: int,
    n_perm: int,
    seed: int,
    outcome: str = "depressed",
    skip_null_perms: bool = False,
) -> dict[str, Any]:
    Xn, y, nmf_kept = _matrix(table, nmf_cols, outcome)
    Xp, _, pv_kept = _matrix(table, pv_cols, outcome)
    Xn, nmf_kept, _ = drop_constants(Xn, nmf_kept)
    Xp, pv_kept, _ = drop_constants(Xp, pv_kept)

    pv_res = nested_loocv_logistic(Xp, y, pv_kept, select_top_k=min(k_pv, Xp.shape[1]))
    pv_perm = {"permutation_p_auc": None}
    if not skip_null_perms:
        pv_perm = permutation_pvalue_nested_logistic(
            Xp, y, pv_kept, select_top_k=min(k_pv, Xp.shape[1]),
            observed_auc=pv_res.get("loocv_auc"),
            n_perm=n_perm, seed=seed,
        )

    nmf_res = nested_loocv_logistic(Xn, y, nmf_kept, select_top_k=min(k_ash, Xn.shape[1]))
    nmf_perm = {"permutation_p_auc": None}
    if not skip_null_perms:
        nmf_perm = permutation_pvalue_nested_logistic(
            Xn, y, nmf_kept, select_top_k=min(k_ash, Xn.shape[1]),
            observed_auc=nmf_res.get("loocv_auc"),
            n_perm=n_perm, seed=seed + 11,
        )

    int_res = nested_loocv_logistic_block(
        Xn, Xp, nmf_kept, pv_kept, y, k_ash=k_ash, k_pv=k_pv,
    )
    int_perm = {"permutation_p_auc": None}
    if not skip_null_perms:
        int_perm = permutation_pvalue_nested_logistic_block(
            Xn, Xp, nmf_kept, pv_kept, y,
            k_ash=k_ash, k_pv=k_pv,
            observed_auc=int_res.get("loocv_auc"),
            n_perm=n_perm, seed=seed + 22,
        )

    delta = permutation_pvalue_nested_integrated_vs_pv(
        Xn, Xp, nmf_kept, pv_kept, y,
        k_ash=k_ash, k_pv=k_pv,
        n_perm=n_perm, seed=seed + 33,
    )

    return {
        "n_participants": len(table),
        "n_depressed": int(y.sum()),
        "nmf_features_pool": len(nmf_kept),
        "pv_features_pool": len(pv_kept),
        "pv_only": {
            "auc": pv_res.get("loocv_auc"),
            "perm_p_vs_null": pv_perm.get("permutation_p_auc"),
            "example_fold_features": pv_res.get("example_fold_features"),
        },
        "nmf_only": {
            "auc": nmf_res.get("loocv_auc"),
            "perm_p_vs_null": nmf_perm.get("permutation_p_auc"),
            "example_fold_features": nmf_res.get("example_fold_features"),
        },
        "integrated_block": {
            "k_pv": k_pv,
            "k_ash": k_ash,
            "auc": int_res.get("loocv_auc"),
            "perm_p_vs_null": int_perm.get("permutation_p_auc"),
            "example_fold_features": int_res.get("example_fold_features"),
        },
        "paired_delta_integrated_minus_pv": {
            "delta_auc": delta.get("observed_delta"),
            "perm_p_delta": delta.get("permutation_p_delta"),
            "bootstrap_delta_ci95": delta.get("bootstrap_delta_ci95"),
            "null_delta_mean": delta.get("null_delta_mean"),
            "auc_integrated": delta.get("observed_auc_integrated"),
            "auc_pv": delta.get("observed_auc_pv"),
        },
    }


def format_report(payload: dict[str, Any]) -> str:
    lines = [
        "PTI validation analyses (reviewer-requested)",
        "=" * 72,
    ]
    for label, nmf in payload.get("nmf_runs", {}).items():
        lines.append(f"\nNMF ({label}): preprocess={nmf['preprocess']}, k={nmf['k']}")
        for t in nmf["themes"]:
            art = f"  ARTIFACT: {t['artifact_tokens']}" if t["artifact_tokens"] else ""
            lines.append(f"  {t['theme']} ({t['name']}): {t['top_words']}{art}")

    for label, models in payload.get("model_results", {}).items():
        lines.append(f"\nModels using NMF variant: {label}")
        lines.append(f"  PV-only nested top-{models['integrated_block']['k_pv']} AUC: "
                     f"{models['pv_only']['auc']}  (perm p vs null: {models['pv_only']['perm_p_vs_null']})")
        lines.append(f"  NMF-only nested top-{models['integrated_block']['k_ash']} AUC: "
                     f"{models['nmf_only']['auc']}  (perm p vs null: {models['nmf_only']['perm_p_vs_null']})")
        lines.append(f"  Integrated {models['integrated_block']['k_pv']}+{models['integrated_block']['k_ash']} AUC: "
                     f"{models['integrated_block']['auc']}  (perm p vs null: {models['integrated_block']['perm_p_vs_null']})")
        d = models["paired_delta_integrated_minus_pv"]
        lines.append(f"  PAIRED ΔAUC (integrated − PV): {d['delta_auc']}")
        lines.append(f"    permutation p (one-sided): {d['perm_p_delta']}")
        lines.append(f"    bootstrap 95% CI: {d['bootstrap_delta_ci95']}")
        lines.append(f"    null Δ mean: {d['null_delta_mean']}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=UNITS_JSONL,
        help="Transcript JSONL (default: bundle demo; use ashley_pv_units.jsonl for real data)",
    )
    ap.add_argument(
        "--subcorpus",
        default=None,
        help="Subcorpus filter, e.g. NepalStandStrong for real Nepal interviews",
    )
    ap.add_argument(
        "--outcomes",
        type=Path,
        default=PIPELINE.parent.parent / "final_ashleys_pipeline" / "data" / "participant_outcomes.csv",
        help="Participant outcomes CSV with include_in_regression and depressed columns",
    )
    ap.add_argument(
        "--patient-table",
        type=Path,
        default=PIPELINE.parent.parent / "nepal_stage2_sensitivity_patient_table.csv",
        help="Mother-level table with pvsub_* columns (from cluster TAB-PO run)",
    )
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--k-pv", type=int, default=5)
    ap.add_argument("--k-ash", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-nmf", action="store_true", help="Skip NMF reruns; use existing output files")
    ap.add_argument(
        "--skip-null-perms",
        action="store_true",
        help="Skip single-stream label-shuffled null tests; run paired ΔAUC only",
    )
    args = ap.parse_args()

    if not args.patient_table.is_file():
        raise SystemExit(
            f"Patient table not found: {args.patient_table}\n"
            "Need pvsub_* columns from tabpo_predictions_nepal.json (cluster export)."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "patient_table": str(args.patient_table.resolve()),
        "n_perm": args.n_perm,
        "seed": args.seed,
        "k_pv": args.k_pv,
        "k_ash": args.k_ash,
        "nmf_runs": {},
        "model_results": {},
    }

    variants = [
        ("default", "default", "default"),
        ("contractions", "contractions", "contractions"),
    ]
    for label, preprocess, tag in variants:
        if args.skip_nmf:
            summary_path = OUTPUT_DIR / f"derived_themes_{tag}_summary.json"
            if not summary_path.is_file():
                raise SystemExit(f"Missing {summary_path}; run NMF first or drop --skip-nmf")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            nmf_info = {
                "preprocess": preprocess,
                "output_tag": tag,
                "summary_path": str(summary_path),
                "participant_csv": str(OUTPUT_DIR / f"derived_themes_{tag}_participant.csv"),
                "themes": _theme_rows_from_summary(summary),
                "k": summary.get("k"),
                "mean_coherence": summary.get("mean_coherence"),
            }
        else:
            nmf_info = run_nmf_variant(
                k=args.k,
                preprocess=preprocess,
                output_tag=tag,
                seed=args.seed,
                input_jsonl=args.input,
                subcorpus=args.subcorpus,
                outcomes=args.outcomes,
            )
        payload["nmf_runs"][label] = nmf_info

        table, nmf_cols, pv_cols = load_merged_table(
            args.patient_table,
            Path(nmf_info["participant_csv"]),
        )
        payload["model_results"][label] = evaluate_models(
            table, nmf_cols, pv_cols,
            k_pv=args.k_pv, k_ash=args.k_ash,
            n_perm=args.n_perm, seed=args.seed,
            skip_null_perms=args.skip_null_perms,
        )

    report = format_report(payload)
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report)
    print(f"\nWrote: {OUT_TXT}")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
