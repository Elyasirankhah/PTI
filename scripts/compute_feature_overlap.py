#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent.parent
DEFAULT_TABLE = PIPELINE / "outputs" / "integrated_patient_table.csv"
OUT_TXT = PIPELINE / "outputs" / "feature_overlap.txt"
OUT_JSON = PIPELINE / "outputs" / "feature_overlap.json"


def nmf_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("T") and c.endswith("_density")]


def pv_cols(df: pd.DataFrame, mode: str) -> list[str]:
    if mode == "subcode":
        return [c for c in df.columns if c.startswith("pvsub_")]
    if mode == "legacy":
        return [c for c in df.columns if c.startswith("dens_pv")]
    # auto: prefer raw subcodes, else legacy groups
    sub = [c for c in df.columns if c.startswith("pvsub_")]
    if sub:
        return sub
    return [c for c in df.columns if c.startswith("dens_pv")]


def drop_constant(cols: list[str], df: pd.DataFrame) -> list[str]:
    kept = []
    for c in cols:
        s = df[c].astype(float)
        if s.nunique(dropna=True) > 1:
            kept.append(c)
    return kept


def mean_abs_cross_corr(df: pd.DataFrame, left: list[str], right: list[str]) -> dict:
    rs: list[float] = []
    pairs: list[tuple[str, str, float]] = []
    for a in left:
        for b in right:
            r = df[a].astype(float).corr(df[b].astype(float))
            if pd.notna(r):
                ar = abs(float(r))
                rs.append(ar)
                pairs.append((a, b, ar))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return {
        "n_pairs": len(rs),
        "mean_abs_r": float(np.mean(rs)) if rs else float("nan"),
        "max_abs_r": float(np.max(rs)) if rs else float("nan"),
        "top_pairs": [{"pv": a, "nmf": b, "abs_r": round(r, 4)} for a, b, r in pairs[:10]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument(
        "--pv-mode",
        choices=("auto", "subcode", "legacy"),
        default="auto",
        help="PV columns: pvsub_* (regression default), dens_pv* legacy, or auto",
    )
    ap.add_argument("--out-txt", type=Path, default=OUT_TXT)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    nmf = drop_constant(nmf_cols(df), df)
    pv = drop_constant(pv_cols(df, args.pv_mode), df)
    pv_mode_used = args.pv_mode
    if args.pv_mode == "auto":
        pv_mode_used = "subcode" if any(c.startswith("pvsub_") for c in pv) else "legacy"

    lines = [
        "Demo feature overlap (PV x NMF mother-level densities)",
        f"Table: {args.table}",
        f"Mothers: {len(df)}",
        f"PV mode: {pv_mode_used}",
        f"NMF features (non-constant): {len(nmf)}",
        f"PV features (non-constant): {len(pv)}",
    ]

    if not pv:
        lines.extend([
            "",
            "ERROR: No non-constant PV columns.",
            "Regenerate the table on compute environment:",
            "  bash run_integrated_model.py 2000 5,4",
            "Requires: outputs/pv_predictions.json",
        ])
        text = "\n".join(lines)
        print(text)
        args.out_txt.write_text(text + "\n", encoding="utf-8")
        return

    stats = mean_abs_cross_corr(df, pv, nmf)
    lines.extend([
        f"Cross-stream pairs: {stats['n_pairs']}",
        f"mean |r| = {stats['mean_abs_r']:.3f}",
        f"max  |r| = {stats['max_abs_r']:.3f}",
        "",
        "Top 10 |r| pairs:",
    ])
    for row in stats["top_pairs"]:
        lines.append(f"  {row['abs_r']:.3f}  {row['pv']}  x  {row['nmf']}")

    text = "\n".join(lines)
    print(text)

    payload = {
        "table": str(args.table),
        "n_mothers": len(df),
        "pv_mode": pv_mode_used,
        "nmf_features": nmf,
        "pv_features": pv,
        **stats,
    }
    args.out_txt.write_text(text + "\n", encoding="utf-8")
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_txt}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
