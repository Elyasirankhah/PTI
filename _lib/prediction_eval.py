from __future__ import annotations

from typing import Any

import numpy as np


def _require_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ImportError(f"sklearn and numpy required: {e}") from e
    return LogisticRegression, accuracy_score, roc_auc_score, LeaveOneOut, StandardScaler


def drop_saturated_features(
    X: np.ndarray,
    cols: list[str],
    y: np.ndarray | None = None,
    *,
    max_prevalence: float = 0.9,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Drop binary/density features present in > max_prevalence of participants."""
    keep_idx = []
    dropped = []
    n = X.shape[0]
    for j, col in enumerate(cols):
        col_vals = X[:, j]
        if np.all(col_vals == col_vals[0]):
            dropped.append(col)
            continue
        if col.endswith("_bin"):
            prev = float(np.mean(col_vals > 0))
        else:
            prev = float(np.mean(col_vals > 0))
        if prev >= max_prevalence or prev <= (1 - max_prevalence):
            dropped.append(col)
            continue
        keep_idx.append(j)
    if not keep_idx:
        return X, cols, dropped
    kept = [cols[i] for i in keep_idx]
    return X[:, keep_idx], kept, dropped


def loocv_classifier(
    X: np.ndarray,
    y: np.ndarray,
    make_clf,
    *,
    needs_scaling: bool = True,
) -> dict[str, Any]:
    """Generic LOOCV for any sklearn classifier with predict_proba or decision_function."""
    _, accuracy_score, roc_auc_score, LeaveOneOut, StandardScaler = _require_sklearn()

    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=int)
    probs = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        clf = make_clf()
        clf.fit(X_train, y[train_idx])
        preds[test_idx[0]] = int(clf.predict(X_test)[0])
        if hasattr(clf, "predict_proba"):
            probs[test_idx[0]] = float(clf.predict_proba(X_test)[0, 1])
        else:
            scores = clf.decision_function(X_test)
            probs[test_idx[0]] = float(scores[0])

    acc = float(accuracy_score(y, preds))
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = None

    return {
        "loocv_accuracy": round(acc, 4),
        "loocv_accuracy_pct": round(100 * acc, 1),
        "loocv_auc": round(auc, 4) if auc is not None else None,
        "predictions": preds.tolist(),
        "probabilities": probs.tolist(),
    }


def loocv_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    l1: bool = False,
) -> dict[str, Any]:
    LogisticRegression, _, _, _, _ = _require_sklearn()

    penalty = "l1" if l1 else "l2"
    solver = "liblinear" if l1 else "lbfgs"

    def make_clf():
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=C,
            penalty=penalty,
            solver=solver,
        )

    return loocv_classifier(X, y, make_clf, needs_scaling=True)


def ml_classifier_specs(n: int) -> list[tuple[str, Any, bool]]:
    """
    LOOCV classifier factories: (name, factory, needs_scaling).

    Models: logistic L2, logistic L1, random forest, RBF SVM, k-NN.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    knn_k = max(3, min(5, n // 4))
    return [
        (
            "logistic_l2",
            lambda: LogisticRegression(
                max_iter=3000, class_weight="balanced", C=1.0, penalty="l2", solver="lbfgs"
            ),
            True,
        ),
        (
            "logistic_l1",
            lambda: LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=1.0,
                penalty="l1",
                solver="liblinear",
            ),
            True,
        ),
        (
            "random_forest",
            lambda: RandomForestClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
            ),
            False,
        ),
        (
            "svm_rbf",
            lambda: SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
            True,
        ),
        (
            f"knn_k{knn_k}",
            lambda k=knn_k: KNeighborsClassifier(n_neighbors=k, weights="distance"),
            True,
        ),
    ]


def compare_ml_models_loocv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compare classifiers on a fixed feature matrix (no nested selection)."""
    specs = ml_classifier_specs(len(y))

    rows: list[dict[str, Any]] = []
    for name, factory, scale in specs:
        try:
            res = loocv_classifier(X, y, factory, needs_scaling=scale)
            rows.append(
                {
                    "model": name,
                    "loocv_accuracy": res["loocv_accuracy"],
                    "loocv_accuracy_pct": res["loocv_accuracy_pct"],
                    "loocv_auc": res["loocv_auc"],
                }
            )
        except Exception as exc:
            rows.append({"model": name, "error": str(exc)})

    rows.sort(
        key=lambda r: (
            r.get("loocv_auc") if r.get("loocv_auc") is not None else -1,
            r.get("loocv_accuracy") or -1,
        ),
        reverse=True,
    )
    if feature_names:
        for r in rows:
            r["features"] = feature_names
    return rows


def _feature_direction(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Compare raw feature means in outcome-positive vs outcome-negative participants."""
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {
            "mean_positive": None,
            "mean_negative": None,
            "direction": "unknown",
        }
    mean_pos = float(np.mean(pos))
    mean_neg = float(np.mean(neg))
    if mean_pos > mean_neg:
        direction = "higher in cases"
    elif mean_pos < mean_neg:
        direction = "lower in cases"
    else:
        direction = "no difference"
    return {
        "mean_positive": round(mean_pos, 4),
        "mean_negative": round(mean_neg, 4),
        "direction": direction,
    }


def univariate_screen(
    X: np.ndarray,
    y: np.ndarray,
    cols: list[str],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """LOOCV logistic on each feature alone; rank by AUC then accuracy."""
    rows = univariate_screen_all(X, y, cols)
    return rows[:top_k]


def univariate_screen_all(
    X: np.ndarray,
    y: np.ndarray,
    cols: list[str],
) -> list[dict[str, Any]]:
    """LOOCV logistic on every feature alone; include direction vs outcome."""
    rows: list[dict[str, Any]] = []
    for j, col in enumerate(cols):
        xj = X[:, j]
        if np.all(xj == xj[0]):
            continue
        res = loocv_logistic(xj.reshape(-1, 1), y, C=1.0, l1=False)
        direction = _feature_direction(xj, y)
        rows.append(
            {
                "feature": col,
                "loocv_accuracy": res["loocv_accuracy"],
                "loocv_auc": res["loocv_auc"],
                "mean_positive": direction["mean_positive"],
                "mean_negative": direction["mean_negative"],
                "direction": direction["direction"],
            }
        )
    rows.sort(
        key=lambda r: (
            r["loocv_auc"] if r["loocv_auc"] is not None else -1,
            r["loocv_accuracy"],
        ),
        reverse=True,
    )
    return rows


def select_l1_features(
    X: np.ndarray,
    y: np.ndarray,
    cols: list[str],
    *,
    c_grid: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """Pick C by LOOCV AUC; return non-zero L1 coefficients from full-data fit."""
    LogisticRegression, _, roc_auc_score, _, StandardScaler = _require_sklearn()

    best_c = c_grid[0]
    best_auc = -1.0
    for c in c_grid:
        res = loocv_logistic(X, y, C=c, l1=True)
        auc = res["loocv_auc"]
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_c = c

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=best_c,
        penalty="l1",
        solver="liblinear",
    )
    clf.fit(Xs, y)
    selected = [cols[i] for i, w in enumerate(clf.coef_[0]) if abs(w) > 1e-8]

    if selected:
        idx = [cols.index(c) for c in selected]
        loocv = loocv_logistic(X[:, idx], y, C=best_c, l1=False)
    else:
        loocv = {"loocv_accuracy": None, "loocv_auc": None}

    return {
        "best_C": best_c,
        "selected_features": selected,
        "loocv_accuracy": loocv.get("loocv_accuracy"),
        "loocv_auc": loocv.get("loocv_auc"),
    }


def permutation_pvalue(
    X: np.ndarray,
    y: np.ndarray,
    *,
    observed_auc: float | None,
    n_perm: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    if observed_auc is None:
        return {"permutation_p_auc": None, "n_perm": n_perm}

    rng = np.random.default_rng(seed)
    null_aucs = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        res = loocv_logistic(X, y_perm, C=1.0, l1=False)
        if res["loocv_auc"] is not None:
            null_aucs.append(res["loocv_auc"])

    if not null_aucs:
        return {"permutation_p_auc": None, "n_perm": n_perm}

    p = float(np.mean(np.array(null_aucs) >= observed_auc))
    return {
        "permutation_p_auc": round(p, 4),
        "null_auc_mean": round(float(np.mean(null_aucs)), 4),
        "n_perm": n_perm,
    }


def permutation_pvalue_auc_difference(
    X_primary: np.ndarray,
    X_baseline: np.ndarray,
    y: np.ndarray,
    *,
    observed_auc_primary: float | None = None,
    observed_auc_baseline: float | None = None,
    n_perm: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    One-sided permutation test: primary LOOCV AUC > baseline LOOCV AUC.

    Shuffles labels and re-fits both models each iteration (paired null for
    'does expanded/gap representation rank better than PV-only?').
    """
    res_p = loocv_logistic(X_primary, y, C=1.0, l1=False)
    res_b = loocv_logistic(X_baseline, y, C=1.0, l1=False)
    auc_p = observed_auc_primary if observed_auc_primary is not None else res_p.get("loocv_auc")
    auc_b = observed_auc_baseline if observed_auc_baseline is not None else res_b.get("loocv_auc")
    if auc_p is None or auc_b is None:
        return {
            "observed_auc_primary": auc_p,
            "observed_auc_baseline": auc_b,
            "observed_auc_delta": None,
            "permutation_p_delta": None,
            "n_perm": n_perm,
        }

    observed_delta = float(auc_p) - float(auc_b)
    rng = np.random.default_rng(seed)
    null_deltas: list[float] = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        rp = loocv_logistic(X_primary, y_perm, C=1.0, l1=False)
        rb = loocv_logistic(X_baseline, y_perm, C=1.0, l1=False)
        if rp["loocv_auc"] is not None and rb["loocv_auc"] is not None:
            null_deltas.append(float(rp["loocv_auc"]) - float(rb["loocv_auc"]))

    if not null_deltas:
        return {
            "observed_auc_primary": round(float(auc_p), 4),
            "observed_auc_baseline": round(float(auc_b), 4),
            "observed_auc_delta": round(observed_delta, 4),
            "permutation_p_delta": None,
            "n_perm": n_perm,
        }

    p = float(np.mean(np.array(null_deltas) >= observed_delta))
    return {
        "observed_auc_primary": round(float(auc_p), 4),
        "observed_auc_baseline": round(float(auc_b), 4),
        "observed_auc_delta": round(observed_delta, 4),
        "permutation_p_delta": round(p, 4),
        "null_delta_mean": round(float(np.mean(null_deltas)), 4),
        "n_perm": n_perm,
        "primary_loocv": {
            "probabilities": res_p.get("probabilities"),
            "predictions": res_p.get("predictions"),
        },
        "baseline_loocv": {
            "probabilities": res_b.get("probabilities"),
            "predictions": res_b.get("predictions"),
        },
    }


def fisher_combine_pvalues(p_values: list[float]) -> dict[str, Any]:
    """Fisher's method to combine independent p-values across sites/tests."""
    try:
        from scipy.stats import chi2
    except ImportError as e:
        raise ImportError("scipy required for fisher_combine_pvalues") from e

    ps = [float(p) for p in p_values if p is not None and not np.isnan(p)]
    if not ps:
        return {"fisher_p": None, "k": 0}
    k = len(ps)
    stat = -2.0 * sum(np.log(max(p, 1e-15)) for p in ps)
    combined = float(chi2.sf(stat, 2 * k))
    return {
        "fisher_p": round(combined, 4),
        "k": k,
        "input_p_values": [round(p, 4) for p in ps],
    }


def bootstrap_auc_difference_ci(
    y: np.ndarray,
    probs_primary: list[float],
    probs_baseline: list[float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap 95% CI for AUC(primary) - AUC(baseline) on paired LOOCV probabilities."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p_primary = np.asarray(probs_primary, dtype=float)
    p_baseline = np.asarray(probs_baseline, dtype=float)
    n = len(y)
    deltas: list[float] = []

    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", category=UserWarning)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            y_b = y[idx]
            if len(np.unique(y_b)) < 2:
                continue
            try:
                auc_p = float(roc_auc_score(y_b, p_primary[idx]))
                auc_b = float(roc_auc_score(y_b, p_baseline[idx]))
                deltas.append(auc_p - auc_b)
            except ValueError:
                pass

    if not deltas:
        return {"n_boot": n_boot, "delta_ci95": None}

    return {
        "n_boot": n_boot,
        "observed_delta": round(float(roc_auc_score(y, p_primary) - roc_auc_score(y, p_baseline)), 4),
        "delta_ci95": [
            round(float(np.percentile(deltas, 2.5)), 4),
            round(float(np.percentile(deltas, 97.5)), 4),
        ],
    }


def bootstrap_ci(
    y: np.ndarray,
    preds: list[int],
    probs: list[float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    import warnings

    from sklearn.metrics import accuracy_score, roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y)
    accs = []
    aucs = []
    y = np.asarray(y)
    preds = np.asarray(preds)
    probs = np.asarray(probs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            accs.append(float(accuracy_score(y[idx], preds[idx])))
            try:
                aucs.append(float(roc_auc_score(y[idx], probs[idx])))
            except ValueError:
                pass

    out: dict[str, Any] = {
        "n_boot": n_boot,
        "accuracy_ci95": [
            round(float(np.percentile(accs, 2.5)), 4),
            round(float(np.percentile(accs, 97.5)), 4),
        ],
    }
    if aucs:
        out["auc_ci95"] = [
            round(float(np.percentile(aucs, 2.5)), 4),
            round(float(np.percentile(aucs, 97.5)), 4),
        ]
    return out


def evaluate_feature_set(
    rows: list[dict],
    outcome_col: str,
    cols: list[str],
    *,
    max_prevalence: float = 0.9,
    top_k: int = 5,
    n_perm: int = 500,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Full small-n evaluation pipeline for one feature column list."""
    if len(rows) < 5 or not cols:
        return {"error": f"too few rows ({len(rows)}) or no features"}

    y = np.array([int(float(r[outcome_col])) for r in rows])
    X = np.array([[float(r.get(c) or 0) for c in cols] for r in rows])

    X_f, cols_f, dropped_sat = drop_saturated_features(X, cols, y, max_prevalence=max_prevalence)
    if X_f.shape[1] == 0:
        return {"error": "all features saturated or constant", "dropped_saturated": dropped_sat}

    uni = univariate_screen(X_f, y, cols_f, top_k=top_k)
    top_feats = [u["feature"] for u in uni]
    top_idx = [cols_f.index(f) for f in top_feats]

    full = loocv_logistic(X_f, y, C=1.0, l1=False)
    topk = loocv_logistic(X_f[:, top_idx], y, C=1.0, l1=False) if top_idx else {}
    l1 = select_l1_features(X_f, y, cols_f)
    ml_models = (
        compare_ml_models_loocv(X_f[:, top_idx], y, feature_names=top_feats)
        if top_idx
        else []
    )

    perm = permutation_pvalue(
        X_f[:, top_idx] if top_idx else X_f,
        y,
        observed_auc=topk.get("loocv_auc"),
        n_perm=n_perm,
        seed=seed,
    )
    if n_boot > 0:
        boot = bootstrap_ci(
            y,
            topk.get("predictions", full["predictions"]),
            topk.get("probabilities", full["probabilities"]),
            n_boot=n_boot,
            seed=seed,
        )
    else:
        boot = {}

    return {
        "n_participants": len(rows),
        "n_features_input": len(cols),
        "n_features_after_saturation_filter": len(cols_f),
        "dropped_saturated": dropped_sat,
        "multivariate_all_features": {
            "loocv_accuracy": full["loocv_accuracy"],
            "loocv_auc": full["loocv_auc"],
            "note": "Not recommended for n=29 with many features; prefer top-k or L1",
        },
        "univariate_top": uni,
        "top_k_model": {
            "k": len(top_feats),
            "features": top_feats,
            "loocv_accuracy": topk.get("loocv_accuracy"),
            "loocv_auc": topk.get("loocv_auc"),
        },
        "l1_selected": l1,
        "ml_models_top_k": ml_models,
        "permutation_on_top_k": perm,
        "bootstrap_ci_top_k": boot,
    }


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


def _pick_threshold_on_train(y_train: np.ndarray, p_train: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy on training fold."""
    if len(np.unique(y_train)) < 2:
        return 0.5
    best_t = 0.5
    best_ba = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p_train >= t).astype(int)
        ba = _balanced_accuracy(y_train, pred)
        if ba > best_ba:
            best_ba = ba
            best_t = float(t)
    return best_t


def _univariate_rank_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cols: list[str],
    *,
    top_k: int,
) -> list[int]:
    """Rank features by train-fold AUC; return column indices."""
    from sklearn.metrics import roc_auc_score

    scores: list[tuple[float, int]] = []
    for j in range(X_train.shape[1]):
        xj = X_train[:, j]
        if np.all(xj == xj[0]):
            continue
        try:
            auc = float(roc_auc_score(y_train, xj))
        except ValueError:
            auc = 0.0
        scores.append((abs(auc - 0.5), j))
    scores.sort(reverse=True)
    return [j for _, j in scores[:top_k]]


def nested_loocv_penalized(
    X: np.ndarray,
    y: np.ndarray,
    cols: list[str],
    *,
    penalty: str = "l2",
    C: float = 1.0,
    l1_ratio: float = 0.5,
) -> dict[str, Any]:
    """
    Nested LOOCV penalized logistic on all supplied features (no full-data screen).

    penalty: 'l2' (ridge) or 'elasticnet'.
    """
    LogisticRegression, _, _, _, _ = _require_sklearn()

    if penalty == "l2":
        def make_clf():
            return LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                C=C,
                penalty="l2",
                solver="lbfgs",
            )
    elif penalty == "elasticnet":
        def make_clf():
            return LogisticRegression(
                max_iter=8000,
                class_weight="balanced",
                C=C,
                penalty="elasticnet",
                l1_ratio=l1_ratio,
                solver="saga",
            )
    else:
        raise ValueError(f"unsupported penalty: {penalty}")

    res = loocv_classifier(X, y, make_clf, needs_scaling=True)
    return {
        "model": f"logistic_{penalty}",
        "penalty": penalty,
        "C": C,
        "l1_ratio": l1_ratio if penalty == "elasticnet" else None,
        "n_features": X.shape[1],
        "features": cols,
        "loocv_accuracy": res["loocv_accuracy"],
        "loocv_accuracy_pct": res["loocv_accuracy_pct"],
        "loocv_auc": res["loocv_auc"],
        "predictions": res["predictions"],
        "probabilities": res["probabilities"],
        "note": "Sensitivity only: all pool themes after saturation filter; no top-k screen",
    }


def nested_loocv_classifier(
    X: np.ndarray,
    y: np.ndarray,
    cols: list[str],
    make_clf,
    *,
    needs_scaling: bool = True,
    feature_indices: list[int] | None = None,
    select_top_k: int | None = None,
    tune_threshold: bool = False,
) -> dict[str, Any]:
    """
    Nested LOOCV with any sklearn classifier: per-fold top-k on train only,
    then fit and predict the held-out participant.
    """
    _, accuracy_score, roc_auc_score, LeaveOneOut, StandardScaler = _require_sklearn()

    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=int)
    probs = np.zeros(len(y), dtype=float)
    fold_features: list[list[str]] = []

    for train_idx, test_idx in loo.split(X):
        X_train_full, X_test_full = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        if feature_indices is not None:
            use_idx = feature_indices
        elif select_top_k is not None:
            use_idx = _univariate_rank_train(
                X_train_full, y_train, cols, top_k=select_top_k
            )
            if not use_idx:
                use_idx = list(range(min(select_top_k, X.shape[1])))
        else:
            use_idx = list(range(X.shape[1]))

        fold_features.append([cols[i] for i in use_idx])
        X_train = X_train_full[:, use_idx]
        X_test = X_test_full[:, use_idx]

        if needs_scaling:
            scaler = StandardScaler()
            X_train_f = scaler.fit_transform(X_train)
            X_test_f = scaler.transform(X_test)
        else:
            X_train_f, X_test_f = X_train, X_test

        clf = make_clf()
        clf.fit(X_train_f, y_train)
        if hasattr(clf, "predict_proba"):
            p_test = float(clf.predict_proba(X_test_f)[0, 1])
        else:
            p_test = float(clf.decision_function(X_test_f)[0])
        probs[test_idx[0]] = p_test

        if tune_threshold and hasattr(clf, "predict_proba"):
            p_train = clf.predict_proba(X_train_f)[:, 1]
            t = _pick_threshold_on_train(y_train, p_train)
            preds[test_idx[0]] = int(p_test >= t)
        else:
            preds[test_idx[0]] = int(clf.predict(X_test_f)[0])

    acc = float(accuracy_score(y, preds))
    ba = _balanced_accuracy(y, preds)
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = None
    majority = max(float(np.mean(y == 0)), float(np.mean(y == 1)))

    return {
        "loocv_accuracy": round(acc, 4),
        "loocv_accuracy_pct": round(100 * acc, 1),
        "balanced_accuracy": round(ba, 4),
        "balanced_accuracy_pct": round(100 * ba, 1),
        "loocv_auc": round(auc, 4) if auc is not None else None,
        "majority_baseline_pct": round(100 * majority, 1),
        "predictions": preds.tolist(),
        "probabilities": probs.tolist(),
        "example_fold_features": fold_features[0] if fold_features else [],
    }


def nested_loocv_logistic(
    X: np.ndarray,
    y: np.ndarray,
    cols: list[str],
    *,
    feature_indices: list[int] | None = None,
    select_top_k: int | None = None,
    tune_threshold: bool = False,
) -> dict[str, Any]:
    """Nested LOOCV with L2 logistic regression (default primary model)."""
    LogisticRegression, _, _, _, _ = _require_sklearn()

    def make_clf():
        return LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0, penalty="l2", solver="lbfgs"
        )

    return nested_loocv_classifier(
        X, y, cols, make_clf,
        needs_scaling=True,
        feature_indices=feature_indices,
        select_top_k=select_top_k,
        tune_threshold=tune_threshold,
    )


def nested_loocv_classifier_block(
    X_ash: np.ndarray,
    X_pv: np.ndarray,
    ash_cols: list[str],
    pv_cols: list[str],
    y: np.ndarray,
    make_clf,
    *,
    needs_scaling: bool = True,
    k_ash: int,
    k_pv: int,
) -> dict[str, Any]:
    """
    Nested LOOCV: top-k per block on train only, concatenate, one classifier.
    """
    _, accuracy_score, roc_auc_score, LeaveOneOut, StandardScaler = _require_sklearn()

    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=int)
    probs = np.zeros(len(y), dtype=float)
    fold_features: list[list[str]] = []

    for train_idx, test_idx in loo.split(y):
        Xa_train, Xa_test = X_ash[train_idx], X_ash[test_idx]
        Xp_train, Xp_test = X_pv[train_idx], X_pv[test_idx]
        y_train = y[train_idx]

        ash_idx = _univariate_rank_train(Xa_train, y_train, ash_cols, top_k=min(k_ash, X_ash.shape[1]))
        pv_idx = _univariate_rank_train(Xp_train, y_train, pv_cols, top_k=min(k_pv, X_pv.shape[1]))
        if not ash_idx:
            ash_idx = list(range(min(k_ash, X_ash.shape[1])))
        if not pv_idx:
            pv_idx = list(range(min(k_pv, X_pv.shape[1])))

        fold_features.append([ash_cols[i] for i in ash_idx] + [pv_cols[j] for j in pv_idx])
        X_train = np.hstack([Xa_train[:, ash_idx], Xp_train[:, pv_idx]])
        X_test = np.hstack([Xa_test[:, ash_idx], Xp_test[:, pv_idx]])

        if needs_scaling:
            scaler = StandardScaler()
            X_train_f = scaler.fit_transform(X_train)
            X_test_f = scaler.transform(X_test)
        else:
            X_train_f, X_test_f = X_train, X_test

        clf = make_clf()
        clf.fit(X_train_f, y_train)
        if hasattr(clf, "predict_proba"):
            p_test = float(clf.predict_proba(X_test_f)[0, 1])
        else:
            p_test = float(clf.decision_function(X_test_f)[0])
        probs[test_idx[0]] = p_test
        preds[test_idx[0]] = int(clf.predict(X_test_f)[0])

    acc = float(accuracy_score(y, preds))
    ba = _balanced_accuracy(y, preds)
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = None
    majority = max(float(np.mean(y == 0)), float(np.mean(y == 1)))

    return {
        "loocv_accuracy": round(acc, 4),
        "loocv_accuracy_pct": round(100 * acc, 1),
        "balanced_accuracy": round(ba, 4),
        "loocv_auc": round(auc, 4) if auc is not None else None,
        "majority_baseline_pct": round(100 * majority, 1),
        "predictions": preds.tolist(),
        "probabilities": probs.tolist(),
        "example_fold_features": fold_features[0] if fold_features else [],
        "k_ash": k_ash,
        "k_pv": k_pv,
        "n_features_per_fold": min(k_ash, X_ash.shape[1]) + min(k_pv, X_pv.shape[1]),
    }


def nested_loocv_logistic_block(
    X_ash: np.ndarray,
    X_pv: np.ndarray,
    ash_cols: list[str],
    pv_cols: list[str],
    y: np.ndarray,
    *,
    k_ash: int,
    k_pv: int,
) -> dict[str, Any]:
    """Nested block LOOCV with L2 logistic regression."""
    LogisticRegression, _, _, _, _ = _require_sklearn()

    def make_clf():
        return LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0, penalty="l2", solver="lbfgs"
        )

    return nested_loocv_classifier_block(
        X_ash, X_pv, ash_cols, pv_cols, y, make_clf,
        needs_scaling=True, k_ash=k_ash, k_pv=k_pv,
    )


def permutation_pvalue_nested_logistic_block(
    X_ash: np.ndarray,
    X_pv: np.ndarray,
    ash_cols: list[str],
    pv_cols: list[str],
    y: np.ndarray,
    *,
    k_ash: int,
    k_pv: int,
    observed_auc: float | None = None,
    n_perm: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    if observed_auc is None:
        observed_auc = nested_loocv_logistic_block(
            X_ash, X_pv, ash_cols, pv_cols, y, k_ash=k_ash, k_pv=k_pv
        ).get("loocv_auc")
    if observed_auc is None:
        return {"permutation_p_auc": None, "n_perm": n_perm}

    rng = np.random.default_rng(seed)
    null_aucs: list[float] = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        res = nested_loocv_logistic_block(
            X_ash, X_pv, ash_cols, pv_cols, y_perm, k_ash=k_ash, k_pv=k_pv
        )
        if res.get("loocv_auc") is not None:
            null_aucs.append(float(res["loocv_auc"]))

    if not null_aucs:
        return {"permutation_p_auc": None, "n_perm": n_perm}

    p = float(np.mean(np.array(null_aucs) >= float(observed_auc)))
    return {
        "permutation_p_auc": round(p, 4),
        "null_auc_mean": round(float(np.mean(null_aucs)), 4),
        "n_perm": n_perm,
        "observed_auc": round(float(observed_auc), 4),
    }


def permutation_pvalue_nested_integrated_vs_pv(
    X_ash: np.ndarray,
    X_pv: np.ndarray,
    ash_cols: list[str],
    pv_cols: list[str],
    y: np.ndarray,
    *,
    k_ash: int = 4,
    k_pv: int = 5,
    n_perm: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Paired permutation test: nested integrated block AUC > nested PV-only AUC.

    Each permutation shuffles labels once and refits both full nested pipelines.
    """
    res_int = nested_loocv_logistic_block(
        X_ash, X_pv, ash_cols, pv_cols, y, k_ash=k_ash, k_pv=k_pv,
    )
    res_pv = nested_loocv_logistic(X_pv, y, pv_cols, select_top_k=k_pv)
    auc_int = res_int.get("loocv_auc")
    auc_pv = res_pv.get("loocv_auc")
    if auc_int is None or auc_pv is None:
        return {
            "observed_auc_integrated": auc_int,
            "observed_auc_pv": auc_pv,
            "observed_delta": None,
            "permutation_p_delta": None,
            "n_perm": n_perm,
        }

    observed_delta = float(auc_int) - float(auc_pv)
    rng = np.random.default_rng(seed)
    null_deltas: list[float] = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        ri = nested_loocv_logistic_block(
            X_ash, X_pv, ash_cols, pv_cols, y_perm, k_ash=k_ash, k_pv=k_pv,
        )
        rp = nested_loocv_logistic(X_pv, y_perm, pv_cols, select_top_k=k_pv)
        if ri.get("loocv_auc") is not None and rp.get("loocv_auc") is not None:
            null_deltas.append(float(ri["loocv_auc"]) - float(rp["loocv_auc"]))

    p = float(np.mean(np.array(null_deltas) >= observed_delta)) if null_deltas else None
    boot = bootstrap_auc_difference_ci(
        y,
        res_int.get("probabilities") or [],
        res_pv.get("probabilities") or [],
        n_boot=min(2000, max(500, n_perm)),
        seed=seed + 1,
    )
    return {
        "observed_auc_integrated": round(float(auc_int), 4),
        "observed_auc_pv": round(float(auc_pv), 4),
        "observed_delta": round(observed_delta, 4),
        "permutation_p_delta": round(p, 4) if p is not None else None,
        "null_delta_mean": round(float(np.mean(null_deltas)), 4) if null_deltas else None,
        "bootstrap_delta_ci95": boot.get("delta_ci95"),
        "n_perm": n_perm,
        "k_ash": k_ash,
        "k_pv": k_pv,
        "integrated_example_fold_features": res_int.get("example_fold_features"),
        "pv_example_fold_features": res_pv.get("example_fold_features"),
        "integrated_probabilities": res_int.get("probabilities"),
        "pv_probabilities": res_pv.get("probabilities"),
    }


def permutation_pvalue_nested_logistic(
    X: np.ndarray,
    y: np.ndarray,
    cols: list[str],
    *,
    select_top_k: int = 5,
    observed_auc: float | None = None,
    n_perm: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Permutation p for nested LOOCV: shuffle labels, rerun fold-wise top-k + logistic each time."""
    if observed_auc is None:
        observed_auc = nested_loocv_logistic(
            X, y, cols, select_top_k=select_top_k
        ).get("loocv_auc")
    if observed_auc is None:
        return {"permutation_p_auc": None, "n_perm": n_perm}

    rng = np.random.default_rng(seed)
    null_aucs: list[float] = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        res = nested_loocv_logistic(
            X, y_perm, cols, select_top_k=select_top_k
        )
        if res.get("loocv_auc") is not None:
            null_aucs.append(float(res["loocv_auc"]))

    if not null_aucs:
        return {"permutation_p_auc": None, "n_perm": n_perm}

    p = float(np.mean(np.array(null_aucs) >= float(observed_auc)))
    return {
        "permutation_p_auc": round(p, 4),
        "null_auc_mean": round(float(np.mean(null_aucs)), 4),
        "n_perm": n_perm,
        "observed_auc": round(float(observed_auc), 4),
    }


DEMO_CURATED_PV = [
    "pv_CareCoordinationPatient_dens",
    "pv_SharedDecisionPatient_dens",
    "pv_PartnershipProvider_dens",
    "pv_PartnershipPatient_dens",
    "pv_CareCoordinationProvider_dens",
]


def evaluate_demo_optimization(
    rows: list[dict],
    outcome_col: str,
    pv_cols: list[str],
    *,
    max_prevalence: float = 0.9,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run nested LOOCV strategies for Demo depression prediction."""
    if len(rows) < 5 or not pv_cols:
        return {"error": "too few rows or no PV columns"}

    y = np.array([int(float(r[outcome_col])) for r in rows])
    X = np.array([[float(r.get(c) or 0) for c in pv_cols] for r in rows])
    X_f, cols_f, dropped = drop_saturated_features(X, pv_cols, y, max_prevalence=max_prevalence)
    if X_f.shape[1] == 0:
        return {"error": "all features saturated", "dropped_saturated": dropped}

    majority = max(float(np.mean(y == 0)), float(np.mean(y == 1)))
    results: dict[str, Any] = {
        "n_participants": len(rows),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "dropped_saturated": dropped,
        "strategies": {
            "majority_baseline": {
                "loocv_accuracy_pct": round(100 * majority, 1),
                "note": "Always predict majority class (non-depressed)",
            }
        },
    }

    uni = univariate_screen(X_f, y, cols_f, top_k=top_k)
    top_idx = [cols_f.index(u["feature"]) for u in uni]

    runs: list[tuple[str, dict[str, Any]]] = [
        ("top3_picked_on_full_data", {"feature_indices": top_idx}),
        ("nested_top3_per_fold", {"select_top_k": top_k}),
        ("nested_top3_tuned_threshold", {"select_top_k": top_k, "tune_threshold": True}),
    ]

    curated_idx = [j for j, c in enumerate(cols_f) if c in DEMO_CURATED_PV]
    if curated_idx:
        runs.append(("curated_pv_codes", {"feature_indices": curated_idx}))
        runs.append(
            ("curated_pv_tuned_threshold", {"feature_indices": curated_idx, "tune_threshold": True})
        )

    for name, kwargs in runs:
        if kwargs.get("feature_indices") is not None and not kwargs["feature_indices"]:
            results["strategies"][name] = {"error": "no features available"}
            continue
        idx = kwargs.get("feature_indices")
        if idx is not None:
            res = nested_loocv_logistic(
                X_f[:, idx],
                y,
                [cols_f[i] for i in idx],
                tune_threshold=kwargs.get("tune_threshold", False),
            )
        else:
            res = nested_loocv_logistic(
                X_f,
                y,
                cols_f,
                select_top_k=kwargs.get("select_top_k"),
                tune_threshold=kwargs.get("tune_threshold", False),
            )
        results["strategies"][name] = res

    return results
