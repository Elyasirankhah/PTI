from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF_KWARGS = {
    "max_features": 3000,
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.85,
    "stop_words": "english",
    "token_pattern": r"[a-zA-Z][a-zA-Z'-]{2,}",
}

NMF_KWARGS = {"max_iter": 500, "init": "nndsvda"}


def infer_k_bounds(n_paragraphs: int) -> tuple[int, int]:
    """Search bounds from corpus size only (no literature window).

    - k_min=2: smallest meaningful factorization rank.
    - k_max=min(floor(sqrt(n_docs)), floor(n_docs/50)): keep topics from becoming
      too sparse while scaling with corpus size.
    """
    k_min = 2
    if n_paragraphs < 4:
        return k_min, k_min
    doc_cap = int(np.floor(np.sqrt(n_paragraphs)))
    span_cap = max(k_min + 1, n_paragraphs // 50)
    k_max = max(k_min + 1, min(doc_cap, span_cap))
    return k_min, k_max


def _top_terms_for_components(components: np.ndarray, terms: np.ndarray, top_n: int) -> list[set[str]]:
    topics: list[set[str]] = []
    for t in range(components.shape[0]):
        idx = components[t].argsort()[::-1][:top_n]
        topics.append({str(terms[i]) for i in idx if components[t, i] > 0})
    return topics


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def _best_topic_alignment_score(topics_a: list[set[str]], topics_b: list[set[str]]) -> float:
    """Mean best-match Jaccard score between two topic lists.

    Uses Hungarian assignment if SciPy is available; otherwise falls back to greedy
    matching (still deterministic).
    """
    k = min(len(topics_a), len(topics_b))
    if k == 0:
        return 0.0
    sim = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            sim[i, j] = _jaccard(topics_a[i], topics_b[j])

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        # maximize similarity => minimize negative similarity
        row_ind, col_ind = linear_sum_assignment(-sim)
        return float(sim[row_ind, col_ind].mean())
    except Exception:
        # Greedy: repeatedly pick best remaining pair
        used_i: set[int] = set()
        used_j: set[int] = set()
        scores: list[float] = []
        for _ in range(k):
            best = (-1.0, -1, -1)
            for i in range(k):
                if i in used_i:
                    continue
                for j in range(k):
                    if j in used_j:
                        continue
                    s = sim[i, j]
                    if s > best[0]:
                        best = (float(s), i, j)
            s, i, j = best
            if i < 0 or j < 0:
                break
            used_i.add(i)
            used_j.add(j)
            scores.append(s)
        return float(np.mean(scores)) if scores else 0.0


def _bootstrap_topics(
    X,
    terms: np.ndarray,
    *,
    k: int,
    seed: int,
    n_bootstrap: int,
    top_n_terms: int,
) -> list[list[set[str]]]:
    rng = np.random.default_rng(seed)
    n_docs = X.shape[0]
    topics_runs: list[list[set[str]]] = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_docs, size=n_docs)  # bootstrap sample
        Xb = X[idx]
        model = NMF(n_components=k, random_state=seed + b, **NMF_KWARGS)
        model.fit(Xb)
        topics_runs.append(_top_terms_for_components(model.components_, terms, top_n_terms))
    return topics_runs


def _stability_score(topics_runs: list[list[set[str]]]) -> tuple[float, float]:
    """Return (mean, std) stability across all bootstrap run pairs."""
    if len(topics_runs) < 2:
        return 0.0, 0.0
    scores: list[float] = []
    for i in range(len(topics_runs)):
        for j in range(i + 1, len(topics_runs)):
            scores.append(_best_topic_alignment_score(topics_runs[i], topics_runs[j]))
    return float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0


def select_k_by_stability(
    texts: list[str],
    seed: int,
    *,
    k_values: list[int] | None = None,
    n_bootstrap: int = 10,
    top_n_terms: int = 15,
) -> dict[str, Any]:
    """Select k by maximizing topic stability across bootstrap samples (unsupervised).

    Procedure:
    - Fit TF-IDF once on the full corpus (fixed vocabulary).
    - For each k in grid (default inferred from n, or explicit k_values):
        * Fit NMF on B bootstrap resamples of documents
        * Extract top-N terms per topic
        * Score pairwise stability as mean optimal-alignment Jaccard similarity
    - Choose k with maximum mean stability (tie-break: smaller k).
    """
    n_docs = len(texts)
    if k_values is not None:
        k_list = sorted({k for k in k_values if k >= 2 and k < n_docs})
        if not k_list:
            k_list = [2]
        k_min, k_max = k_list[0], k_list[-1]
    else:
        k_min, k_max = infer_k_bounds(n_docs)
        k_list = list(range(k_min, k_max + 1))

    vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    print(f"  Stability selection: k={k_min}..{k_max}, bootstrap={n_bootstrap}, seed={seed}")
    print(f"  Vocabulary size m={len(terms)}, n={n_docs}, top_terms={top_n_terms}")

    grid: list[dict[str, Any]] = []
    for k in k_list:
        print(f"  k={k} ...", end=" ", flush=True)
        runs = _bootstrap_topics(
            X,
            terms,
            k=k,
            seed=seed,
            n_bootstrap=n_bootstrap,
            top_n_terms=top_n_terms,
        )
        mean_s, std_s = _stability_score(runs)
        grid.append(
            {
                "k": k,
                "stability_mean_jaccard": round(mean_s, 4),
                "stability_std_jaccard": round(std_s, 4),
            }
        )
        print(f"stability={mean_s:.4f} (+/- {std_s:.4f})")

    best = max(grid, key=lambda r: (r["stability_mean_jaccard"], -r["k"]))
    # tie-break: smaller k if equal stability
    best_stab = best["stability_mean_jaccard"]
    tol = max(1e-9, abs(best_stab) * 1e-4)
    tied = [r for r in grid if r["stability_mean_jaccard"] >= best_stab - tol]
    chosen = min(tied, key=lambda r: r["k"])

    return {
        "chosen_k": chosen["k"],
        "chosen_stability_mean_jaccard": chosen["stability_mean_jaccard"],
        "chosen_stability_std_jaccard": chosen["stability_std_jaccard"],
        "method": "bootstrap topic stability (optimal alignment Jaccard on top terms)",
        "selection_criterion": "argmax_k mean_pairwise_stability (tie-break: smaller k)",
        "k_bounds_inferred": {"k_min": k_min, "k_max": k_max, "k_values": k_list},
        "seed": seed,
        "n_paragraphs": n_docs,
        "n_bootstrap": n_bootstrap,
        "top_n_terms": top_n_terms,
        "grid": grid,
    }


def format_selection_report(selection: dict[str, Any]) -> str:
    bounds = selection["k_bounds_inferred"]
    lines = [
        "NMF k selection (model-based)",
        "=" * 60,
        f"Method: {selection['method']}",
        f"Criterion: {selection['selection_criterion']}",
        f"Search window: k={bounds['k_min']}..{bounds['k_max']} (from n={selection['n_paragraphs']} paragraphs)",
        f"Bootstrap: {selection.get('n_bootstrap', '?')}, top_terms: {selection.get('top_n_terms', '?')}, seed: {selection['seed']}",
        "",
        f"Selected k: {selection['chosen_k']} "
        f"(stability={selection['chosen_stability_mean_jaccard']:.4f} +/- {selection['chosen_stability_std_jaccard']:.4f})",
        "",
        f"{'k':>3}  {'stability':>10}  {'std':>8}",
    ]
    for r in selection["grid"]:
        mark = " *" if r["k"] == selection["chosen_k"] else ""
        lines.append(
            f"{r['k']:3d}  {r['stability_mean_jaccard']:10.4f}  {r['stability_std_jaccard']:8.4f}{mark}"
        )
    return "\n".join(lines) + "\n"
