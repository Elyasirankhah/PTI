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

from _lib.filters import is_junk_paragraph, is_participant_turn  # noqa: E402
from _lib.io_utils import load_jsonl  # noqa: E402
from _lib.participants import participant_id_from_source  # noqa: E402
from _lib.sites import DEMO_SUBCORPUS  # noqa: E402
from _lib.nmf_k_selection import format_selection_report, select_k_by_stability  # noqa: E402
from _lib.outcomes import load_regression_pids  # noqa: E402
from config import OUTPUT_DIR, PARTICIPANT_OUTCOMES, UNITS_JSONL  # noqa: E402

OUT_DIR = OUTPUT_DIR
PARA_CSV = OUT_DIR / "derived_themes.csv"
SUMMARY_TXT = OUT_DIR / "derived_themes_summary.txt"
SUMMARY_JSON = OUT_DIR / "derived_themes_summary.json"
PARTICIPANT_CSV = OUT_DIR / "derived_themes_participant.csv"

_SPEAKER_PREFIX = re.compile(
    r"^(P|Participant|Respondent|R)\s*[:;.\-–—]+\s*", re.I
)
_MIN_WORDS = 5

# Primary k=6 theme names (interpretive labels from top words + quote review)
THEME_LABELS_K6: dict[str, str] = {
    "T0": "Baby care and health",
    "T1": "Emotional difficulty and help-seeking",
    "T2": "Activities and preferences",
    "T3": "Family and in-laws",
    "T4": "Denial of problems and negation talk",
    "T5": "Daily chores and hygiene",
}


def theme_display_name(theme_label: str, k: int) -> str:
    if k == 6 and theme_label in THEME_LABELS_K6:
        return THEME_LABELS_K6[theme_label]
    return ""


def strip_speaker(text: str) -> str:
    return _SPEAKER_PREFIX.sub("", (text or "").strip()).strip("- ").strip()


_CONTRACTION_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\bdon't\b", "do not"),
    (r"\bdont\b", "do not"),
    (r"\bdidn't\b", "did not"),
    (r"\bdidnt\b", "did not"),
    (r"\bcan't\b", "cannot"),
    (r"\bcant\b", "cannot"),
    (r"\bwon't\b", "will not"),
    (r"\bwont\b", "will not"),
    (r"\bisn't\b", "is not"),
    (r"\bisnt\b", "is not"),
    (r"\bwasn't\b", "was not"),
    (r"\bwasnt\b", "was not"),
    (r"\bhasn't\b", "has not"),
    (r"\bhasnt\b", "has not"),
    (r"\bhaven't\b", "have not"),
    (r"\bhavent\b", "have not"),
    (r"\bwouldn't\b", "would not"),
    (r"\bwouldnt\b", "would not"),
    (r"\bshouldn't\b", "should not"),
    (r"\bshouldnt\b", "should not"),
    (r"\bcouldn't\b", "could not"),
    (r"\bcouldnt\b", "could not"),
    (r"\baren't\b", "are not"),
    (r"\barent\b", "are not"),
    (r"\bweren't\b", "were not"),
    (r"\bwerent\b", "were not"),
)


def normalize_contractions(text: str) -> str:
    """Expand common contractions before tokenization (reduces don/didn artifacts)."""
    t = (text or "").replace("\u2019", "'").replace("`", "'")
    for pattern, repl in _CONTRACTION_REPLACEMENTS:
        t = re.sub(pattern, repl, t, flags=re.I)
    return t


def clean_text(text: str, *, preprocess: str = "default") -> str:
    t = strip_speaker(text)
    if preprocess == "contractions":
        t = normalize_contractions(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z]{2,}", text))


# ── Load & filter ──────────────────────────────────────────────────────

def load_demo_participant_paragraphs(
    jsonl_path: Path,
    *,
    allowed_pids: set[str] | None = None,
    preprocess: str = "default",
    subcorpus: str | None = None,
) -> list[dict[str, Any]]:
    """Return strict participant paragraphs with cleaned text."""
    subcorpus_filter = subcorpus or DEMO_SUBCORPUS
    rows = load_jsonl(jsonl_path)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("subcorpus") != subcorpus_filter:
            continue
        pid = participant_id_from_source(row.get("source") or "").upper()
        if allowed_pids is not None and pid not in allowed_pids:
            continue
        if not is_participant_turn(row, strict=True):
            continue
        raw = (row.get("context") or "").strip()
        cleaned = clean_text(raw, preprocess=preprocess)
        if word_count(cleaned) < _MIN_WORDS:
            continue
        out.append({
            "record_id": row.get("record_id", ""),
            "participant_id": pid,
            "source": row.get("source", ""),
            "paragraph_index": row.get("paragraph_index", 0),
            "text_raw": raw,
            "text_clean": cleaned,
        })
    return out


# ── NMF topic modeling ─────────────────────────────────────────────────

def run_nmf(
    paragraphs: list[dict[str, Any]],
    *,
    k: int,
    seed: int,
    max_features: int = 3000,
    top_n_words: int = 15,
) -> dict[str, Any]:
    """Run TF-IDF + NMF, return themes with top words and assignments."""
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [p["text_clean"] for p in paragraphs]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        stop_words="english",
        token_pattern=r"[a-zA-Z][a-zA-Z'-]{2,}",
    )
    tfidf = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    model = NMF(
        n_components=k,
        random_state=seed,
        max_iter=500,
        init="nndsvda",
    )
    W = model.fit_transform(tfidf)  # (n_docs, k) — paragraph-topic weights
    H = model.components_           # (k, n_features) — topic-word weights

    reconstruction_error = model.reconstruction_err_

    themes: list[dict[str, Any]] = []
    for topic_idx in range(k):
        top_word_idx = H[topic_idx].argsort()[::-1][:top_n_words]
        top_words = [(str(terms[i]), float(H[topic_idx, i])) for i in top_word_idx if H[topic_idx, i] > 0]

        top_para_idx = W[:, topic_idx].argsort()[::-1][:5]
        examples = []
        for pi in top_para_idx:
            if W[pi, topic_idx] > 0.01:
                examples.append({
                    "record_id": paragraphs[pi]["record_id"],
                    "participant_id": paragraphs[pi]["participant_id"],
                    "weight": round(float(W[pi, topic_idx]), 4),
                    "text": paragraphs[pi]["text_clean"][:300],
                })

        themes.append({
            "theme_id": topic_idx,
            "theme_label": f"T{topic_idx}",
            "top_words": top_words,
            "top_word_summary": ", ".join(w for w, _ in top_words[:8]),
            "n_primary_paragraphs": int(np.sum(W[:, topic_idx].argmax() == topic_idx) if k > 1 else len(texts)),
            "examples": examples,
        })

    dominant_topic = W.argmax(axis=1)
    for i, theme in enumerate(themes):
        theme["n_primary_paragraphs"] = int(np.sum(dominant_topic == i))

    assignments: list[dict[str, Any]] = []
    for idx, para in enumerate(paragraphs):
        weights = {f"T{t}": round(float(W[idx, t]), 4) for t in range(k)}
        dominant = int(dominant_topic[idx])
        assignments.append({
            **para,
            "dominant_theme": f"T{dominant}",
            "dominant_weight": round(float(W[idx, dominant]), 4),
            **weights,
        })

    coherence = _compute_coherence(texts, terms, H, top_n=10)

    return {
        "k": k,
        "seed": seed,
        "n_paragraphs": len(paragraphs),
        "reconstruction_error": round(float(reconstruction_error), 4),
        "mean_coherence": round(float(coherence), 4),
        "themes": themes,
        "assignments": assignments,
    }


def _compute_coherence(
    texts: list[str], terms: np.ndarray, H: np.ndarray, top_n: int = 10
) -> float:
    """UMass-style coherence: avg log(D(w_i, w_j) + 1 / D(w_j)) over top words."""
    doc_term = defaultdict(set)
    for idx, text in enumerate(texts):
        lower = text.lower()
        for j, term in enumerate(terms):
            if term in lower:
                doc_term[term].add(idx)

    n_docs = len(texts)
    scores = []
    for topic_idx in range(H.shape[0]):
        top_idx = H[topic_idx].argsort()[::-1][:top_n]
        top_terms = [str(terms[i]) for i in top_idx if H[topic_idx, i] > 0]
        topic_scores = []
        for i in range(1, len(top_terms)):
            for j in range(i):
                d_ij = len(doc_term[top_terms[i]] & doc_term[top_terms[j]])
                d_j = len(doc_term[top_terms[j]])
                if d_j > 0:
                    topic_scores.append(np.log((d_ij + 1) / d_j))
        if topic_scores:
            scores.append(float(np.mean(topic_scores)))
    return float(np.mean(scores)) if scores else 0.0


# ── Participant-level densities ────────────────────────────────────────

def build_participant_densities(
    assignments: list[dict[str, Any]], k: int
) -> list[dict[str, Any]]:
    """Roll up paragraph assignments to participant-level theme densities."""
    by_pid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in assignments:
        by_pid[a["participant_id"]].append(a)

    rows = []
    for pid in sorted(by_pid):
        paras = by_pid[pid]
        n = len(paras)
        theme_counts: Counter[str] = Counter()
        theme_weight_sums: dict[str, float] = defaultdict(float)
        for p in paras:
            theme_counts[p["dominant_theme"]] += 1
            for t in range(k):
                theme_weight_sums[f"T{t}"] += p.get(f"T{t}", 0.0)

        row: dict[str, Any] = {
            "participant_id": pid,
            "n_paragraphs": n,
        }
        for t in range(k):
            label = f"T{t}"
            row[f"{label}_count"] = theme_counts.get(label, 0)
            row[f"{label}_density"] = round(theme_counts.get(label, 0) / n, 4) if n else 0
            row[f"{label}_weight_mean"] = round(theme_weight_sums[label] / n, 4) if n else 0
        rows.append(row)
    return rows


# ── k scan ─────────────────────────────────────────────────────────────

def scan_k_range(
    paragraphs: list[dict[str, Any]],
    k_min: int,
    k_max: int,
    seed: int,
) -> list[dict[str, Any]]:
    results = []
    for k in range(k_min, k_max + 1):
        print(f"  k={k} ...", end=" ", flush=True)
        res = run_nmf(paragraphs, k=k, seed=seed)
        results.append({
            "k": k,
            "reconstruction_error": res["reconstruction_error"],
            "mean_coherence": res["mean_coherence"],
        })
        print(f"recon_err={res['reconstruction_error']:.4f}  coherence={res['mean_coherence']:.4f}")
    return results


def auto_select_k(
    paragraphs: list[dict[str, Any]],
    seed: int,
    *,
    n_bootstrap: int = 10,
    top_n_terms: int = 15,
) -> dict[str, Any]:
    """Pick k by bootstrap topic stability (unsupervised, no manual k)."""
    texts = [p["text_clean"] for p in paragraphs]
    print(f"Auto k: {len(texts)} paragraphs — model selects k by bootstrap stability")
    selection = select_k_by_stability(texts, seed, n_bootstrap=n_bootstrap, top_n_terms=top_n_terms)
    print()
    print(format_selection_report(selection))
    return selection


# ── Output ─────────────────────────────────────────────────────────────

def write_summary_txt(
    result: dict[str, Any],
    path: Path,
    k_scan: list[dict] | None = None,
    *,
    cohort_label: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Demo data-derived themes (Stage 1)",
        "=" * 60,
        f"Cohort: {cohort_label or 'unspecified'}",
        f"Participant paragraphs: {result['n_paragraphs']}",
        f"Number of themes (k): {result['k']}",
        f"Seed: {result['seed']}",
        f"Reconstruction error: {result['reconstruction_error']}",
        f"Mean coherence (UMass): {result['mean_coherence']}",
        "",
        "Method: TF-IDF + NMF on strict participant turns.",
        "No hand-written themes or keyword lists used.",
        "Depression labels NOT used for theme discovery.",
        "",
    ]

    if k_scan:
        lines.append("k scan results:")
        lines.append(f"  {'k':>3s}  {'recon_err':>10s}  {'coherence':>10s}")
        for r in k_scan:
            lines.append(f"  {r['k']:3d}  {r['reconstruction_error']:10.4f}  {r['mean_coherence']:10.4f}")
        best = max(k_scan, key=lambda r: r["mean_coherence"])
        lines.append(f"  Best coherence at k={best['k']} ({best['mean_coherence']:.4f})")
        lines.append("")

    for theme in result["themes"]:
        lines.append("-" * 60)
        name = theme_display_name(theme["theme_label"], result["k"])
        title = f"Theme {theme['theme_label']}: {name}" if name else f"Theme {theme['theme_label']}"
        lines.append(f"{title}")
        lines.append(f"  Top words: {theme['top_word_summary']}")
        lines.append(f"  Primary paragraphs: {theme['n_primary_paragraphs']}")
        lines.append(f"  Weighted top words: {', '.join(f'{w} ({s:.3f})' for w, s in theme['top_words'][:10])}")
        lines.append("  Example quotes:")
        for ex in theme["examples"][:3]:
            lines.append(f"    [{ex['participant_id']}, w={ex['weight']:.3f}]")
            lines.append(f"    \"{ex['text'][:200]}...\"")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Summary: {path}")


def write_summary_json(result: dict[str, Any], path: Path, k_scan: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "method": "TF-IDF + NMF on strict Demo participant turns; no keywords or outcome labels",
        "n_paragraphs": result["n_paragraphs"],
        "k": result["k"],
        "seed": result["seed"],
        "reconstruction_error": result["reconstruction_error"],
        "mean_coherence": result["mean_coherence"],
        "themes": [
            {
                "theme_id": t["theme_id"],
                "theme_label": t["theme_label"],
                "theme_name": theme_display_name(t["theme_label"], result["k"]) or None,
                "top_word_summary": t["top_word_summary"],
                "top_words": t["top_words"],
                "n_primary_paragraphs": t["n_primary_paragraphs"],
                "examples": t["examples"],
            }
            for t in result["themes"]
        ],
    }
    if k_scan:
        obj["k_scan"] = k_scan
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON:    {path}")


def write_para_csv(assignments: list[dict[str, Any]], path: Path, k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "record_id", "participant_id", "source", "paragraph_index",
        "text_clean", "dominant_theme", "dominant_weight",
    ] + [f"T{t}" for t in range(k)]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in assignments:
            w.writerow(row)
    print(f"  Paras:   {path}  ({len(assignments)} rows)")


def write_participant_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Patient: {path}  ({len(rows)} rows)")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: data-derived theme extraction")
    ap.add_argument("--input", type=Path, default=UNITS_JSONL,
                     help="Full transcripts JSONL (default: data/units.jsonl)")
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="Fix k manually (default: model selects k by 5-fold CV reconstruction error)",
    )
    ap.add_argument("--k-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                     help="Scan k from MIN to MAX and pick best in-sample coherence")
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=10,
        help="Bootstrap resamples for automatic k selection (default: 10)",
    )
    ap.add_argument(
        "--top-terms",
        type=int,
        default=15,
        help="Top terms per topic used for stability (default: 15)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-words", type=int, default=15,
                     help="Number of top words per theme (default: 15)")
    ap.add_argument(
        "--cohort",
        choices=("regression", "all"),
        default="regression",
        help="regression=n=29 (9 Depressed + 20 Non-Depressed); all=include TrialParticipants",
    )
    ap.add_argument("--outcomes", type=Path, default=PARTICIPANT_OUTCOMES)
    ap.add_argument(
        "--preprocess",
        choices=("default", "contractions"),
        default="default",
        help="Text preprocessing before TF-IDF (contractions=expand don't/didn't etc.)",
    )
    ap.add_argument(
        "--output-tag",
        default="",
        help="Suffix for output files, e.g. contractions -> derived_themes_contractions_summary.txt",
    )
    ap.add_argument(
        "--subcorpus",
        default=None,
        help=f"Subcorpus filter (default: {DEMO_SUBCORPUS}; use NepalStandStrong for real corpus)",
    )
    args = ap.parse_args()

    allowed_pids = None
    cohort_label = "all Demo interviews"
    if args.cohort == "regression":
        allowed_pids = load_regression_pids(args.outcomes, subcorpus=args.subcorpus)
        cohort_label = f"n=29 regression cohort (9 Depressed + 20 Non-Depressed; TrialParticipants excluded)"

    print(f"Loading transcripts from {args.input} ...")
    print(f"  Cohort: {cohort_label}")
    paragraphs = load_demo_participant_paragraphs(
        args.input,
        allowed_pids=allowed_pids,
        preprocess=args.preprocess,
        subcorpus=args.subcorpus,
    )
    pids = sorted({p["participant_id"] for p in paragraphs})
    print(f"  {len(paragraphs)} strict participant paragraphs from {len(pids)} participants")
    print(f"  Participants: {', '.join(pids)}")
    print()

    k_scan: list[dict] | None = None
    chosen_k = args.k

    if args.k is not None and args.k_range:
        ap.error("Use only one of --k or --k-range")
    if args.k is None and not args.k_range:
        k_selection = auto_select_k(
            paragraphs,
            args.seed,
            n_bootstrap=args.bootstrap,
            top_n_terms=args.top_terms,
        )
        k_scan = k_selection["grid"]
        chosen_k = k_selection["chosen_k"]
        (OUT_DIR / "nmf_k_selection_report.txt").write_text(
            format_selection_report(k_selection),
            encoding="utf-8",
        )
        (OUT_DIR / "nmf_k_selection_report.json").write_text(
            json.dumps(k_selection, indent=2),
            encoding="utf-8",
        )
        print(f"  Wrote k selection report: {OUT_DIR / 'nmf_k_selection_report.txt'}")
    elif args.k_range:
        k_min, k_max = args.k_range
        print(f"Scanning k = {k_min} to {k_max} ...")
        k_scan = scan_k_range(paragraphs, k_min, k_max, args.seed)
        best = max(k_scan, key=lambda r: r["mean_coherence"])
        chosen_k = best["k"]
        print(f"\nBest coherence at k={chosen_k} ({best['mean_coherence']:.4f})")
        print()

    print(f"Running NMF with k={chosen_k}, seed={args.seed}, preprocess={args.preprocess} ...")
    result = run_nmf(paragraphs, k=chosen_k, seed=args.seed, top_n_words=args.top_words)

    tag = f"_{args.output_tag}" if args.output_tag else ""
    summary_txt = OUT_DIR / f"derived_themes{tag}_summary.txt"
    summary_json = OUT_DIR / f"derived_themes{tag}_summary.json"
    para_csv = OUT_DIR / f"derived_themes{tag}.csv"
    participant_csv = OUT_DIR / f"derived_themes{tag}_participant.csv"

    print(f"\n{'=' * 60}")
    print(f"Discovered {chosen_k} themes from {result['n_paragraphs']} paragraphs")
    print(f"Reconstruction error: {result['reconstruction_error']}")
    print(f"Mean coherence: {result['mean_coherence']}")
    print()
    for theme in result["themes"]:
        name = theme_display_name(theme["theme_label"], chosen_k)
        label = f"{theme['theme_label']}: {name}" if name else theme["theme_label"]
        print(f"  {label}")
        print(f"    {theme['top_word_summary']}")
        print(f"    ({theme['n_primary_paragraphs']} primary paragraphs)")
    print()

    print("Writing outputs ...")
    write_summary_txt(result, summary_txt, k_scan, cohort_label=cohort_label)
    write_summary_json(result, summary_json, k_scan)
    write_para_csv(result["assignments"], para_csv, chosen_k)

    densities = build_participant_densities(result["assignments"], chosen_k)
    write_participant_csv(densities, participant_csv)

    print("\nDone. Stage 1 complete — themes extracted from data, no expert lists used.")


if __name__ == "__main__":
    main()
