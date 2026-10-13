#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PIPE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(PIPE / "scripts"))

from _lib.outcomes import load_regression_pids  # noqa: E402
from _lib.participants import participant_id_from_source  # noqa: E402
from _lib.patient_features import PV_GROUP_CODES, PV_SUBCODE_GROUPS  # noqa: E402
from _lib.prediction_eval import (  # noqa: E402
    drop_saturated_features,
    loocv_logistic,
    nested_loocv_logistic,
    permutation_pvalue,
    permutation_pvalue_auc_difference,
    univariate_screen,
)
from _lib.sites import DEMO_SUBCORPUS  # noqa: E402
from config import CODING_SHEET, OUTPUT_DIR, PARTICIPANT_OUTCOMES, UNITS_JSONL  # noqa: E402
from regression_utils import (  # noqa: E402
    load_pv_from_coding_sheet,
    load_pv_from_inline,
    load_pv_from_predictions,
    load_strict_paragraphs,
    safe_col,
)

OUT_TXT = OUTPUT_DIR / "feature_utils.txt"
OUT_JSON = OUTPUT_DIR / "feature_utils.json"
OUT_TABLE = OUTPUT_DIR / "feature_utils_patient_table.csv"


def _out_paths(tag: str | None) -> tuple[Path, Path, Path]:
    if tag:
        safe = tag.replace("/", "_").strip()
        return (
            OUTPUT_DIR / f"feature_utils_{safe}.txt",
            OUTPUT_DIR / f"feature_utils_{safe}.json",
            OUTPUT_DIR / f"feature_utils_patient_table_{safe}.csv",
        )
    return OUT_TXT, OUT_JSON, OUT_TABLE

NMF_PARTICIPANT = OUTPUT_DIR / "derived_themes_participant.csv"

LEGACY_PV_FIXED = [
    "dens_pv_care_coordination",
    "dens_pvsub_care_coordination",
    "pv_CareCoordinationPatient_dens",
    "dens_pvsub_sdoh_economic",
    "dens_pv_partnership",
]


def read_csv(path: Path) -> list[dict[str, str]]:
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


def plain_label(col: str) -> str:
    s = col
    for prefix in ("pvsub_", "pv_", "dens_pvsub_", "dens_pv_", "T", "_density"):
        s = s.replace(prefix, "")
    return s.strip("_").lower() or col


def subcode_match(code: str, sub: str, want_code: str, want_sub: str) -> bool:
    if code != want_code:
        return False
    if want_sub == "*":
        return True
    return sub == want_sub


def build_extended_pv_features(
    paragraphs: list[dict[str, Any]],
    pv_labels: dict[str, list[tuple[str, str]]],
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    """Build code, subcode, and legacy-group PV density columns."""
    pid_n: Counter[str] = Counter()
    pid_code: dict[str, Counter[str]] = defaultdict(Counter)
    pid_sub: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)

    for para in paragraphs:
        pid = participant_id_from_source(para.get("source") or "").upper()
        rid = para.get("record_id", "")
        pid_n[pid] += 1
        for code, sub in pv_labels.get(rid, []):
            pid_code[pid][code] += 1
            pid_sub[pid][(code, sub)] += 1

    pools: dict[str, list[str]] = {
        "pv_code": [],
        "pv_subcode_raw": [],
        "pv_legacy_group": [],
        "pv_legacy_per_code": [],
    }
    all_sub_keys: set[tuple[str, str]] = set()
    for pid in pid_sub:
        all_sub_keys.update(pid_sub[pid].keys())

    code_cols = sorted({f"pv_{safe_col(c)}" for c in {c for pid in pid_code for c in pid_code[pid]}})
    pools["pv_code"] = code_cols

    sub_cols = sorted(
        f"pvsub_{safe_col(c)}_{safe_col(s if s and s != 'None' else 'none')}"
        for c, s in all_sub_keys
    )
    pools["pv_subcode_raw"] = sub_cols

    legacy_group_cols = [f"dens_pv_{g}" for g in PV_GROUP_CODES] + [
        f"dens_pvsub_{g}" for g in PV_SUBCODE_GROUPS
    ]
    pools["pv_legacy_group"] = legacy_group_cols

    per_code_cols = sorted(
        f"pv_{safe_col(c)}_dens"
        for c in {c for pid in pid_code for c in pid_code[pid]}
    )
    pools["pv_legacy_per_code"] = per_code_cols

    densities: dict[str, dict[str, float]] = {}
    for pid, n in pid_n.items():
        d: dict[str, float] = {}
        for code in pid_code[pid]:
            d[f"pv_{safe_col(code)}"] = round(pid_code[pid][code] / n, 6)
            d[f"pv_{safe_col(code)}_dens"] = round(pid_code[pid][code] / n, 6)

        for c, s in pid_sub[pid]:
            col = f"pvsub_{safe_col(c)}_{safe_col(s if s and s != 'None' else 'none')}"
            d[col] = round(pid_sub[pid][(c, s)] / n, 6)

        for gname, codes in PV_GROUP_CODES.items():
            cnt = sum(pid_code[pid].get(c, 0) for c in codes)
            d[f"dens_pv_{gname}"] = round(cnt / n, 6)

        for gname, combos in PV_SUBCODE_GROUPS.items():
            cnt = 0
            for code, sub in combos:
                for (c, s), nlab in pid_sub[pid].items():
                    if subcode_match(c, s, code, sub):
                        cnt += nlab
            d[f"dens_pvsub_{gname}"] = round(cnt / n, 6)

        densities[pid] = d

    return densities, pools


def merge_tables(
    nmf_path: Path,
    pv_densities: dict[str, dict[str, float]],
    pv_pools: dict[str, list[str]],
    outcomes_path: Path,
    outcome_col: str,
) -> list[dict[str, Any]]:
    nmf_rows = read_csv(nmf_path)
    outcomes = {
        r["participant_id"].strip().upper(): r
        for r in read_csv(outcomes_path)
        if r.get("subcorpus") == DEMO_SUBCORPUS
    }
    ash_cols = [c for c in nmf_rows[0] if c.endswith("_density")]
    all_pv_cols = sorted(
        {c for cols in pv_pools.values() for c in cols}
    )

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
        for col in ash_cols:
            row[col] = float(nmf_row.get(col) or 0)
        pv = pv_densities.get(pid, {})
        for col in all_pv_cols:
            row[col] = pv.get(col, 0.0)
        table.append(row)
    return table


def _matrix(table: list[dict[str, Any]], cols: list[str], outcome_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    use = [c for c in cols if c in table[0]]
    y = np.array([int(r[outcome_col]) for r in table])
    X = np.array([[float(r.get(c) or 0) for c in use] for r in table])
    return X, y, use


def drop_constants(X: np.ndarray, cols: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    X_f, kept, dropped = drop_saturated_features(X, cols, max_prevalence=1.0)
    return X_f, kept, dropped


def select_top_k_frequency(table: list[dict[str, Any]], cols: list[str], k: int) -> list[str]:
    ranked: list[tuple[str, float, float]] = []
    for col in cols:
        vals = [float(r.get(col) or 0) for r in table]
        prev = sum(v > 0 for v in vals) / len(vals)
        mean = sum(vals) / len(vals)
        ranked.append((col, prev, mean))
    ranked.sort(key=lambda t: (-t[1], -t[2], t[0]))

    chosen: list[str] = []
    seen: set[str] = set()
    for col, _, _ in ranked:
        label = plain_label(col)
        if label in seen:
            continue
        chosen.append(col)
        seen.add(label)
        if len(chosen) >= k:
            break
    return chosen


def eval_spec_clean(
    table: list[dict[str, Any]],
    feat_cols: list[str],
    outcome_col: str,
    *,
    name: str,
    method: str,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    X_raw, y, cols = _matrix(table, feat_cols, outcome_col)
    if not cols:
        return {"name": name, "method": method, "error": "no features"}

    X, kept, dropped = drop_constants(X_raw, cols)
    if X.shape[1] == 0:
        return {"name": name, "method": method, "error": "all constant", "dropped": dropped}

    used_cols = kept
    univariate_top: list[dict[str, Any]] | None = None
    if method == "full":
        X_use = X
    elif method.startswith("top"):
        k = int(method.replace("top", ""))
        top = univariate_screen(X, y, kept, top_k=min(k, X.shape[1]))
        used_cols = [t["feature"] for t in top]
        idx = [kept.index(c) for c in used_cols]
        X_use = X[:, idx]
        univariate_top = top
    elif method.startswith("freq"):
        k = int(method.replace("freq", ""))
        used_cols = select_top_k_frequency(table, kept, k)
        idx = [kept.index(c) for c in used_cols]
        X_use = X[:, idx]
    elif method.startswith("nested"):
        k = int(method.replace("nested", ""))
        res = nested_loocv_logistic(X, y, kept, select_top_k=min(k, X.shape[1]))
        return {
            "name": name,
            "method": method,
            "features": res.get("example_fold_features"),
            "n_features": min(k, X.shape[1]),
            "dropped_constant": dropped,
            "auc": res.get("loocv_auc"),
            "accuracy": res.get("loocv_accuracy"),
            "perm_p": None,
            "note": "nested LOOCV: top-k selected inside each fold; perm p omitted (too slow)",
        }
    else:
        raise ValueError(method)

    res = loocv_logistic(X_use, y, C=1.0)
    auc = res.get("loocv_auc")
    perm = permutation_pvalue(X_use, y, observed_auc=auc, n_perm=n_perm, seed=seed)
    out: dict[str, Any] = {
        "name": name,
        "method": method,
        "features": used_cols,
        "n_features": len(used_cols),
        "dropped_constant": dropped,
        "auc": auc,
        "accuracy": res.get("loocv_accuracy"),
        "perm_p": perm.get("permutation_p_auc"),
    }
    if univariate_top is not None:
        out["univariate_top"] = [
            {
                "feature": t["feature"],
                "loocv_auc": round(float(t["loocv_auc"]), 4) if t.get("loocv_auc") is not None else None,
            }
            for t in univariate_top
        ]
    return out


def run_grid(
    table: list[dict[str, Any]],
    ash_cols: list[str],
    pv_pools: dict[str, list[str]],
    outcome_col: str,
    *,
    n_perm: int,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    methods = ["full", "top3", "top4", "top5", "freq3", "freq4", "freq5", "nested3", "nested4", "nested5"]

    for method in methods:
        results.append(
            eval_spec_clean(
                table, ash_cols, outcome_col,
                name=f"NMF theme ({method})", method=method,
                n_perm=n_perm, seed=seed,
            )
        )

    for pool_name, cols in pv_pools.items():
        present = [c for c in cols if c in table[0]]
        for method in methods:
            results.append(
                eval_spec_clean(
                    table, present, outcome_col,
                    name=f"PV {pool_name} ({method})", method=method,
                    n_perm=n_perm, seed=seed,
                )
            )

    legacy_cols = [c for c in LEGACY_PV_FIXED if c in table[0]]
    results.append(
        eval_spec_clean(
            table, legacy_cols, outcome_col,
            name="PV legacy fixed-5 bundle (full)", method="full",
            n_perm=n_perm, seed=seed,
        )
    )

    return results


def best_by_family(results: list[dict[str, Any]]) -> dict[str, Any]:
    ash = [r for r in results if r.get("name", "").startswith("NMF") and r.get("auc") is not None]
    pv = [r for r in results if r.get("name", "").startswith("PV") and r.get("auc") is not None]
    ash.sort(key=lambda r: r["auc"], reverse=True)
    pv.sort(key=lambda r: r["auc"], reverse=True)
    return {
        "best_nmf": ash[0] if ash else None,
        "best_pv": pv[0] if pv else None,
    }


def write_report(path: Path, results: list[dict[str, Any]], bests: dict[str, Any], pv_pool_sizes: dict[str, int]) -> None:
    def fmt(value: Any) -> str:
        return "NA" if value is None else str(value)

    lines = [
        "Demo Stage 2 sensitivity: PV subcodes, top-k bundles, legacy fixed-5",
        "=" * 72,
        "Same n=29, same 1,203 strict participant paragraphs, outcome=depressed",
        "",
        "PV pool sizes:",
    ]
    for k, n in pv_pool_sizes.items():
        lines.append(f"  {k}: {n} features")
    lines.append("")
    lines.append(f"{'Model':<42} {'Method':<8} {'AUC':>6} {'p':>7}  Features")
    lines.append("-" * 72)
    for r in sorted(results, key=lambda x: (x.get("auc") or -1), reverse=True):
        if r.get("error"):
            lines.append(f"{r.get('name','?'):<42} {r.get('method','?'):<8} {'—':>6} {'—':>7}  {r['error']}")
            continue
        feats = ", ".join(r.get("features") or [])
        lines.append(
            f"{r['name']:<42} {r['method']:<8} {fmt(r.get('auc')):>6} {fmt(r.get('perm_p')):>7}  {feats}"
        )

    lines.append("")
    lines.append("-" * 72)
    ba = bests.get("best_nmf")
    bp = bests.get("best_pv")
    if ba:
        lines.append(f"Best NMF: {ba['name']}  AUC={ba['auc']}  p={ba.get('perm_p')}")
    if bp:
        lines.append(f"Best PV:     {bp['name']}  AUC={bp['auc']}  p={bp.get('perm_p')}")
    if ba and bp:
        lines.append(f"Delta (PV - NMF): {round(float(bp['auc']) - float(ba['auc']), 4)}")

    top5 = next(
        (
            r for r in results
            if r.get("name") == "PV pv_subcode_raw (top5)" and r.get("univariate_top")
        ),
        None,
    )
    if top5:
        lines.extend([
            "",
            "Univariate LOOCV AUC (each subcode alone) → top-5 combined model above:",
            f"{'Feature':<62} {'AUC':>6}",
            "-" * 72,
        ])
        for row in top5["univariate_top"]:
            lines.append(f"{row['feature']:<62} {row['loocv_auc']:>6}")
        lines.append(f"{'Combined (5 features)':<62} {top5['auc']:>6}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2 sensitivity analyses")
    ap.add_argument("--input", type=Path, default=UNITS_JSONL)
    ap.add_argument("--nmf-patient", type=Path, default=NMF_PARTICIPANT)
    ap.add_argument("--coding-sheet", type=Path, default=CODING_SHEET)
    ap.add_argument("--predictions", type=Path, default=None)
    ap.add_argument("--outcomes", type=Path, default=PARTICIPANT_OUTCOMES)
    ap.add_argument("--outcome", type=str, default="depressed")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--tag",
        default=None,
        help="Output suffix, e.g. 8b -> feature_utils_8b.txt",
    )
    args = ap.parse_args()
    out_txt, out_json, out_table = _out_paths(args.tag)

    pids = load_regression_pids(args.outcomes)
    paragraphs = load_strict_paragraphs(args.input, pids)
    rids = {p["record_id"] for p in paragraphs}
    rows_by_id = {p["record_id"]: p for p in paragraphs}

    if args.predictions and args.predictions.is_file():
        pv_labels = load_pv_from_predictions(args.predictions, rids, rows_by_id)
    else:
        pv_labels = load_pv_from_coding_sheet(args.coding_sheet, rids)
        if not pv_labels:
            pv_labels = load_pv_from_inline(paragraphs, rids)

    pv_densities, pv_pools = build_extended_pv_features(paragraphs, pv_labels)
    table = merge_tables(args.nmf_patient, pv_densities, pv_pools, args.outcomes, args.outcome)
    write_csv(table, out_table)

    ash_cols = [c for c in table[0] if c.startswith("T") and c.endswith("_density")]
    pool_sizes = {k: len([c for c in v if c in table[0]]) for k, v in pv_pools.items()}

    print("Stage 2 sensitivity")
    print(f"  patients={len(table)} paragraphs={len(paragraphs)}")
    print(f"  NMF features={len(ash_cols)}")
    for k, n in pool_sizes.items():
        print(f"  PV {k}: {n}")

    results = run_grid(table, ash_cols, pv_pools, args.outcome, n_perm=args.n_perm, seed=args.seed)
    bests = best_by_family(results)

    # Head-to-head: best PV vs best NMF
    if bests.get("best_nmf") and bests.get("best_pv"):
        ba, bp = bests["best_nmf"], bests["best_pv"]
        Xa, y, ca = _matrix(table, ba["features"], args.outcome)
        Xp, _, cp = _matrix(table, bp["features"], args.outcome)
        diff = permutation_pvalue_auc_difference(Xp, Xa, y, n_perm=args.n_perm, seed=args.seed)
        bests["head_to_head"] = diff

    write_report(out_txt, results, bests, pool_sizes)
    out_json.write_text(
        json.dumps(
            {
                "predictions": str(args.predictions.resolve()) if args.predictions else None,
                "tag": args.tag,
                "results": results,
                "best": bests,
                "pool_sizes": pool_sizes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {out_txt}")
    print(f"Wrote {out_json}")
    if bests.get("best_nmf"):
        print(f"Best NMF: {bests['best_nmf']['name']} AUC={bests['best_nmf']['auc']}")
    if bests.get("best_pv"):
        print(f"Best PV:     {bests['best_pv']['name']} AUC={bests['best_pv']['auc']}")


if __name__ == "__main__":
    main()
