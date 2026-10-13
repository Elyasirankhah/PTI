#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE))

from _lib.filters import is_participant_turn  # noqa: E402
from _lib.io_utils import load_jsonl  # noqa: E402
from _lib.outcomes import load_regression_pids  # noqa: E402
from _lib.participants import participant_id_from_source  # noqa: E402
from _lib.prediction_eval import (  # noqa: E402
    bootstrap_auc_difference_ci,
    bootstrap_ci,
    drop_saturated_features,
    loocv_logistic,
    permutation_pvalue,
    permutation_pvalue_auc_difference,
)
from _lib.sites import DEMO_SUBCORPUS  # noqa: E402
from config import (  # noqa: E402
    CODING_SHEET,
    OUTPUT_DIR,
    PARTICIPANT_OUTCOMES,
    UNITS_JSONL,
)

NMF_PARTICIPANT = OUTPUT_DIR / "derived_themes_participant.csv"
OUT_TABLE = OUTPUT_DIR / "stage2_patient_table.csv"
OUT_RESULTS_TXT = OUTPUT_DIR / "stage2_results.txt"
OUT_RESULTS_JSON = OUTPUT_DIR / "stage2_results.json"

_SPEAKER_PREFIX = re.compile(r"^(P|Participant|Respondent|R)\s*[:;.\-–—]+\s*", re.I)
_MIN_WORDS = 5


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z]{2,}", text))


def clean_text(text: str) -> str:
    t = _SPEAKER_PREFIX.sub("", (text or "").strip()).strip("- ").strip()
    return re.sub(r"\s+", " ", t).strip()


# ── Load the same strict participant paragraphs ───────────────────────

def load_strict_paragraphs(
    jsonl_path: Path,
    allowed_pids: set[str],
) -> list[dict[str, Any]]:
    rows = load_jsonl(jsonl_path)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("subcorpus") != DEMO_SUBCORPUS:
            continue
        pid = participant_id_from_source(row.get("source") or "").upper()
        if pid not in allowed_pids:
            continue
        if not is_participant_turn(row, strict=True):
            continue
        raw = (row.get("context") or "").strip()
        if word_count(clean_text(raw)) < _MIN_WORDS:
            continue
        out.append(row)
    return out


# ── PV label loading ──────────────────────────────────────────────────

def load_pv_from_coding_sheet(
    path: Path,
    valid_rids: set[str],
) -> dict[str, list[tuple[str, str]]]:
    labels: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        if row.get("subcorpus") != DEMO_SUBCORPUS:
            continue
        if (row.get("label_source") or "") != "pv_valid":
            continue
        rid = (row.get("record_id") or "").strip()
        if rid not in valid_rids:
            continue
        code = (row.get("pv_code") or "").strip()
        sub = (row.get("pv_subcode") or "None").strip() or "None"
        if code:
            labels[rid].append((code, sub))
    return labels


def load_pv_from_predictions(
    path: Path,
    valid_rids: set[str],
    rows_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[str, str]]]:
    try:
        from _lib.pv_label import is_valid_pv_span, parse_pv_prediction
    except ImportError:
        print("  WARNING: pv_metrics not available; skipping predictions JSON")
        return {}

    labels: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if not path.is_file():
        return labels
    payload = json.loads(path.read_text(encoding="utf-8"))
    for rid, raw in zip(payload.get("record_ids", []), payload.get("predictions", [])):
        if rid not in valid_rids:
            continue
        row = rows_by_id.get(rid)
        if not row:
            continue
        context = row.get("context") or ""
        for item in parse_pv_prediction(raw):
            span = (item.get("Span") or "").strip()
            if span and span not in context and span.lower() not in context.lower():
                continue
            if is_valid_pv_span(item):
                code = (item.get("Code") or "").strip()
                sub = (item.get("Sub-code") or "None").strip() or "None"
                labels[rid].append((code, sub))
    return labels


def load_pv_from_inline(
    paragraphs: list[dict[str, Any]],
    valid_rids: set[str],
) -> dict[str, list[tuple[str, str]]]:
    """Extract PV labels from the 'answer' field embedded in JSONL rows."""
    try:
        from _lib.pv_label import is_valid_pv_span
    except ImportError:
        return {}
    labels: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in paragraphs:
        rid = row.get("record_id", "")
        if rid not in valid_rids:
            continue
        ans_raw = row.get("answer") or ""
        if not ans_raw:
            continue
        try:
            ans = json.loads(ans_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        results = ans.get("results") or []
        for item in results:
            if is_valid_pv_span(item):
                code = (item.get("Code") or "").strip()
                sub = (item.get("Sub-code") or "None").strip() or "None"
                if code:
                    labels[rid].append((code, sub))
    return labels


# ── Build PV person-level densities ───────────────────────────────────

def safe_col(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def build_pv_person_densities(
    paragraphs: list[dict[str, Any]],
    pv_labels: dict[str, list[tuple[str, str]]],
) -> tuple[dict[str, dict[str, float]], list[str]]:
    pid_paragraphs: dict[str, int] = Counter()
    pid_code_counts: dict[str, Counter[str]] = defaultdict(Counter)
    all_codes: set[str] = set()

    for para in paragraphs:
        pid = participant_id_from_source(para.get("source") or "").upper()
        rid = para.get("record_id", "")
        pid_paragraphs[pid] += 1
        for code, sub in pv_labels.get(rid, []):
            col = f"pv_{safe_col(code)}"
            pid_code_counts[pid][col] += 1
            all_codes.add(col)

    pv_cols = sorted(all_codes)
    densities: dict[str, dict[str, float]] = {}
    for pid, n in pid_paragraphs.items():
        d: dict[str, float] = {}
        for col in pv_cols:
            d[col] = round(pid_code_counts[pid].get(col, 0) / n, 6) if n else 0.0
        densities[pid] = d
    return densities, pv_cols


# ── Merge patient table ───────────────────────────────────────────────

def build_patient_table(
    nmf_path: Path,
    pv_densities: dict[str, dict[str, float]],
    pv_cols: list[str],
    outcomes_path: Path,
    outcome_col: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    nmf_rows = read_csv(nmf_path)
    outcomes = {
        r["participant_id"].strip().upper(): r
        for r in read_csv(outcomes_path)
        if r.get("subcorpus") == DEMO_SUBCORPUS
    }

    nmf_cols = [c for c in nmf_rows[0] if c.endswith("_density")]
    table: list[dict[str, Any]] = []

    for nmf_row in nmf_rows:
        pid = nmf_row["participant_id"].strip().upper()
        oc = outcomes.get(pid, {})
        outcome_val = (oc.get(outcome_col) or "").strip()
        if outcome_val not in {"0", "1"}:
            continue

        row: dict[str, Any] = {
            "participant_id": pid,
            outcome_col: int(outcome_val),
            "n_paragraphs": int(nmf_row.get("n_paragraphs") or 0),
        }
        for col in nmf_cols:
            row[col] = float(nmf_row.get(col) or 0)
        pv = pv_densities.get(pid, {})
        for col in pv_cols:
            row[col] = pv.get(col, 0.0)
        table.append(row)

    return table, nmf_cols, pv_cols


# ── LOOCV comparison ──────────────────────────────────────────────────

def _matrix(
    table: list[dict[str, Any]],
    feat_cols: list[str],
    outcome_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = [c for c in feat_cols if c in table[0]]
    y = np.array([int(r[outcome_col]) for r in table])
    X = np.array([[float(r.get(c) or 0) for c in cols] for r in table])
    return X, y, cols


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
    X_raw, y, used_cols = _matrix(table, feat_cols, outcome_col)
    if X_raw.shape[1] == 0:
        return {
            "name": name,
            "n_features": 0,
            "features": [],
            "auc": None,
            "note": "no features available",
        }

    # max_prevalence=1.0: only drop truly constant columns.
    # NMF densities are continuous (non-zero for most participants), so
    # the binary prevalence filter would incorrectly drop useful features.
    X, kept, dropped = drop_saturated_features(X_raw, used_cols, y, max_prevalence=1.0)
    if X.shape[1] == 0:
        return {
            "name": name,
            "n_features_before_filter": len(used_cols),
            "n_features": 0,
            "features": [],
            "dropped": dropped,
            "auc": None,
            "note": "all features dropped (constant)",
        }

    res = loocv_logistic(X, y, C=1.0)
    auc = res["loocv_auc"]

    perm = permutation_pvalue(
        X, y,
        observed_auc=auc,
        n_perm=n_perm,
        seed=seed,
    )

    ci = bootstrap_ci(
        y,
        res["predictions"],
        res["probabilities"],
        n_boot=n_boot,
        seed=seed,
    )

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    preds = np.array(res["predictions"])
    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    sens = round(tp / n_pos, 4) if n_pos > 0 else 0.0
    spec = round(tn / n_neg, 4) if n_neg > 0 else 0.0

    return {
        "name": name,
        "n_features_before_filter": len(used_cols),
        "n_features": X.shape[1],
        "features": kept,
        "dropped": dropped,
        "auc": round(auc, 4) if auc is not None else None,
        "accuracy": res["loocv_accuracy"],
        "sensitivity": sens,
        "specificity": spec,
        "perm_p": perm.get("permutation_p_auc"),
        "bootstrap_auc_ci95": ci.get("auc_ci95"),
        "bootstrap_acc_ci95": ci.get("accuracy_ci95"),
        "_probabilities": res["probabilities"],
        "_predictions": res["predictions"],
    }


def run_auc_difference(
    table: list[dict[str, Any]],
    nmf_cols: list[str],
    pv_cols: list[str],
    outcome_col: str,
    *,
    n_perm: int = 2000,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    X_a_raw, y, cols_a = _matrix(table, nmf_cols, outcome_col)
    X_p_raw, _, cols_p = _matrix(table, pv_cols, outcome_col)

    if X_a_raw.shape[1] == 0 or X_p_raw.shape[1] == 0:
        return {"note": "cannot compare — one or both feature sets empty"}

    X_a, kept_a, _ = drop_saturated_features(X_a_raw, cols_a, y, max_prevalence=1.0)
    X_p, kept_p, _ = drop_saturated_features(X_p_raw, cols_p, y, max_prevalence=1.0)

    if X_a.shape[1] == 0 or X_p.shape[1] == 0:
        return {"note": "cannot compare — one or both feature sets empty after filter"}

    diff_perm = permutation_pvalue_auc_difference(
        X_p, X_a, y,
        n_perm=n_perm,
        seed=seed,
    )

    res_a = loocv_logistic(X_a, y, C=1.0)
    res_p = loocv_logistic(X_p, y, C=1.0)

    diff_boot = bootstrap_auc_difference_ci(
        y,
        res_p["probabilities"],
        res_a["probabilities"],
        n_boot=n_boot,
        seed=seed,
    )

    return {
        "nmf_auc": diff_perm.get("observed_auc_baseline"),
        "pv_auc": diff_perm.get("observed_auc_primary"),
        "auc_diff_pv_minus_nmf": diff_perm.get("observed_auc_delta"),
        "perm_p_delta": diff_perm.get("permutation_p_delta"),
        "bootstrap_delta_ci95": diff_boot.get("delta_ci95"),
    }


# ── Report ────────────────────────────────────────────────────────────

def write_report(
    path: Path,
    *,
    n: int,
    n_paragraphs: int,
    n_pv_labeled: int,
    outcome_col: str,
    results: list[dict[str, Any]],
    auc_diff: dict[str, Any],
    nmf_cols: list[str],
    pv_cols: list[str],
) -> None:
    lines = [
        "Demo Stage 2: NMF (NMF) vs PV (TAB-PO) regression comparison",
        "=" * 70,
        f"Cohort: n={n} (9 Depressed + 20 Non-Depressed)",
        f"Strict participant paragraphs: {n_paragraphs}",
        f"Paragraphs with >= 1 PV label: {n_pv_labeled}",
        f"Outcome: {outcome_col}",
        "",
        "Design:",
        "  Same people, same paragraphs, same outcome, same LOOCV method.",
        "  Only the feature source changes (NMF theme vs TAB-PO PV).",
        "  No demographics. No hand-written themes.",
        "",
        f"NMF features ({len(nmf_cols)}): {', '.join(nmf_cols)}",
        f"PV features ({len(pv_cols)}): {', '.join(pv_cols) if pv_cols else '(none available locally)'}",
        "",
    ]

    for res in results:
        lines.append("-" * 70)
        lines.append(f"MODEL: {res['name']}")
        if res.get("auc") is None:
            lines.append(f"  {res.get('note', 'no features')}")
            lines.append("")
            continue
        lines.append(f"  Features: {res['n_features']} (of {res['n_features_before_filter']} before prevalence filter)")
        lines.append(f"  Used: {', '.join(res['features'])}")
        if res.get("dropped"):
            lines.append(f"  Dropped (near-constant): {', '.join(res['dropped'])}")
        lines.append(f"  LOOCV AUC:         {res['auc']}")
        lines.append(f"  LOOCV Accuracy:    {res['accuracy']}")
        lines.append(f"  Sensitivity:       {res['sensitivity']}")
        lines.append(f"  Specificity:       {res['specificity']}")
        lines.append(f"  Permutation p:     {res['perm_p']}")
        lines.append(f"  Bootstrap AUC CI:  {res['bootstrap_auc_ci95']}")
        lines.append(f"  Bootstrap Acc CI:  {res['bootstrap_acc_ci95']}")
        lines.append("")

    lines.append("-" * 70)
    lines.append("AUC DIFFERENCE: PV vs NMF")
    if "note" in auc_diff:
        lines.append(f"  {auc_diff['note']}")
    else:
        lines.append(f"  NMF AUC:         {auc_diff.get('nmf_auc')}")
        lines.append(f"  PV AUC:             {auc_diff.get('pv_auc')}")
        lines.append(f"  Diff (PV - NMF): {auc_diff.get('auc_diff_pv_minus_nmf')}")
        lines.append(f"  Perm p (delta):     {auc_diff.get('perm_p_delta')}")
        lines.append(f"  Bootstrap 95% CI:   {auc_diff.get('bootstrap_delta_ci95')}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("INTERPRETATION")
    lines.append("  NMF themes = data-derived (NMF) textual patterns.")
    lines.append("  PV codes = ontology-derived TAB-PO constructs.")
    lines.append("  Overlap is expected and informative — it shows where the")
    lines.append("  ontology captures themes already present in narratives.")
    lines.append("  Combined model tests whether PV adds predictive signal")
    lines.append("  beyond data-derived text themes.")
    lines.append("  If PV adds beyond NMF: combined AUC > NMF-only AUC.")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: NMF vs PV regression")
    ap.add_argument("--input", type=Path, default=UNITS_JSONL)
    ap.add_argument("--nmf-patient", type=Path, default=NMF_PARTICIPANT)
    ap.add_argument("--coding-sheet", type=Path, default=CODING_SHEET)
    ap.add_argument("--predictions", type=Path, default=None,
                     help="pv_predictions.json (compute environment)")
    ap.add_argument("--outcomes", type=Path, default=PARTICIPANT_OUTCOMES)
    ap.add_argument("--outcome", type=str, default="depressed")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Stage 2: NMF (NMF) vs PV (TAB-PO) regression comparison")
    print("=" * 60)

    pids = load_regression_pids(args.outcomes)
    print(f"Cohort: {len(pids)} participants")

    print("Loading strict participant paragraphs ...")
    paragraphs = load_strict_paragraphs(args.input, pids)
    rids = {p["record_id"] for p in paragraphs}
    rows_by_id = {p["record_id"]: p for p in paragraphs}
    print(f"  {len(paragraphs)} paragraphs from {len(pids)} participants")

    # ── Load PV labels from all available sources ──
    print("Loading PV labels ...")
    pv_labels: dict[str, list[tuple[str, str]]] = {}

    if args.predictions and args.predictions.is_file():
        print(f"  Source: predictions JSON ({args.predictions})")
        pv_labels = load_pv_from_predictions(args.predictions, rids, rows_by_id)
    else:
        # Try coding sheet first
        cs_labels = load_pv_from_coding_sheet(args.coding_sheet, rids)
        if cs_labels:
            pv_labels = cs_labels
            print(f"  Source: coding sheet ({args.coding_sheet})")
        else:
            # Try inline answer field
            inline = load_pv_from_inline(paragraphs, rids)
            if inline:
                pv_labels = inline
                print("  Source: inline 'answer' field in JSONL")
            else:
                print("  WARNING: No PV labels found locally.")
                print("  PV-only and Combined models will be empty.")
                print("  Run on compute environment with --predictions to get PV data.")

    n_pv_labeled = sum(1 for rid in rids if rid in pv_labels and pv_labels[rid])
    print(f"  {n_pv_labeled} / {len(rids)} paragraphs have >= 1 PV label")

    pv_densities, pv_cols = build_pv_person_densities(paragraphs, pv_labels)
    print(f"  PV feature columns: {len(pv_cols)}")
    if pv_cols:
        print(f"  PV codes found: {', '.join(pv_cols)}")

    print("\nBuilding patient table ...")
    table, nmf_cols, pv_cols = build_patient_table(
        args.nmf_patient, pv_densities, pv_cols, args.outcomes, args.outcome
    )
    n_pos = sum(r[args.outcome] for r in table)
    print(f"  {len(table)} patients, {len(nmf_cols)} NMF cols, {len(pv_cols)} PV cols")
    print(f"  Outcome: {args.outcome} ({n_pos} positive, {len(table) - n_pos} negative)")

    write_csv(table, OUT_TABLE)
    print(f"  Saved: {OUT_TABLE}")

    print(f"\nRunning LOOCV models (n_perm={args.n_perm}, n_boot={args.n_boot}) ...")

    # 1. NMF-only
    print("\n  [1/3] NMF-only ...")
    nmf_res = run_model(
        "NMF-only (NMF themes)", table, nmf_cols, args.outcome,
        n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
    )
    print(f"        AUC = {nmf_res.get('auc', 'N/A')}, p = {nmf_res.get('perm_p', 'N/A')}")

    # 2. PV-only
    print("  [2/3] PV-only ...")
    pv_res = run_model(
        "PV-only (TAB-PO codes)", table, pv_cols, args.outcome,
        n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
    )
    print(f"        AUC = {pv_res.get('auc', 'N/A')}, p = {pv_res.get('perm_p', 'N/A')}")

    # 3. Combined
    print("  [3/3] Combined (NMF + PV) ...")
    combined_cols = nmf_cols + pv_cols
    combined_res = run_model(
        "Combined (NMF + PV)", table, combined_cols, args.outcome,
        n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
    )
    print(f"        AUC = {combined_res.get('auc', 'N/A')}, p = {combined_res.get('perm_p', 'N/A')}")

    # AUC difference
    print("\n  AUC difference test (PV vs NMF) ...")
    if pv_cols:
        auc_diff = run_auc_difference(
            table, nmf_cols, pv_cols, args.outcome,
            n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
        )
    else:
        auc_diff = {"note": "cannot compare — PV feature set empty (no labels locally)"}

    results = [nmf_res, pv_res, combined_res]

    write_report(
        OUT_RESULTS_TXT,
        n=len(table),
        n_paragraphs=len(paragraphs),
        n_pv_labeled=n_pv_labeled,
        outcome_col=args.outcome,
        results=results,
        auc_diff=auc_diff,
        nmf_cols=nmf_cols,
        pv_cols=pv_cols,
    )

    # Strip internal fields before JSON dump
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if not k.startswith("_")}
        json_results.append(jr)

    json_out = {
        "cohort_n": len(table),
        "n_paragraphs": len(paragraphs),
        "n_pv_labeled": n_pv_labeled,
        "outcome": args.outcome,
        "nmf_features": nmf_cols,
        "pv_features": pv_cols,
        "models": {r["name"]: r for r in json_results},
        "auc_difference": auc_diff,
    }
    OUT_RESULTS_JSON.write_text(
        json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nResults: {OUT_RESULTS_TXT}")
    print(f"         {OUT_RESULTS_JSON}")
    print(f"Table:   {OUT_TABLE}")
    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
