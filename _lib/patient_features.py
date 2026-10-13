from __future__ import annotations

from collections import Counter, defaultdict

# PV codes treated as SDOH family for group density
PV_SDOH_CODES = frozenset({"SDOH"})

NMF_GROUP_CODES: dict[str, tuple[str, ...]] = {
    "mental_distress": ("mental_distress", "suicidal_ideation"),
    "caregiving": ("caregiving_parenting",),
    "family_stigma": ("family_conflict", "social_stigma"),
    "religion": ("religion_spirituality",),
    "poverty": ("poverty_economic",),
    "health_access": ("healthcare_access",),
    "help_seeking": ("concealment_help_seeking",),
}

PV_GROUP_CODES: dict[str, tuple[str, ...]] = {
    "sdoh": ("SDOH",),
    "socio_emotional": ("SocioEmotionalBehaviour",),
    "care_coordination": ("CareCoordinationPatient",),
    "partnership": ("PartnershipPatient",),
}

# Fine-grained PV (Code, Sub-code) groups so themes that share a top-level PV
# code (e.g. several SDOH themes) get *distinct* density features.
# Key = feature suffix; value = list of (Code, Sub-code) combos to sum.
# Sub-code "*" is a wildcard: match that Code regardless of sub-code (used when
# TAB-PO emits the Code with a blank/variable sub-code, e.g. SocioEmotional).
# Row-count / burden meta-features added after the original README headline runs.
# They dominate gap-only/combined pools on balanced profiles and mask theme densities.
BURDEN_META_COLS = frozenset(
    {
        "pv_label_rows",
        "pv_burden_unique",
        "gap_label_rows",
        "gap_burden_unique",
        "gap_ratio",
        "total_label_rows",
    }
)

PV_SUBCODE_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "sdoh_health_access": (("SDOH", "HealthCareAccessAndQuality"),),
    "sdoh_economic": (("SDOH", "EconomicStability"),),
    "sdoh_social_community": (("SDOH", "SocialAndCommunityContext"),),
    "sdoh_neighborhood": (("SDOH", "NeighborhoodAndBuiltEnvironment"),),
    "socio_emotional": (("SocioEmotionalBehaviour", "*"),),
    "partnership_express": (("PartnershipPatient", "expressOpinions"),),
    "shared_decision": (("SharedDecisionPatient", "*"),),
    "care_coordination": (("CareCoordinationPatient", "*"),),
}


def _is_gap_label(label_source: str) -> bool:
    src = label_source or ""
    return src.startswith("pv_gap") or src == "paragraph_theme"


def _is_pv_label(label_source: str) -> bool:
    return (label_source or "") == "pv_valid"


def build_participant_feature_rows(
    coding_rows: list[dict],
    *,
    float_fmt: bool = True,
) -> list[dict]:
    """One wide row per participant from paragraph-level coding sheet rows."""
    para_by_pid: dict[str, set[str]] = defaultdict(set)
    pv_cnt: dict[str, Counter] = defaultdict(Counter)
    pv_sub_cnt: dict[str, Counter] = defaultdict(Counter)
    ash_cnt: dict[str, Counter] = defaultdict(Counter)
    pv_rows: dict[str, int] = defaultdict(int)
    gap_rows: dict[str, int] = defaultdict(int)
    meta: dict[str, dict] = {}

    for row in coding_rows:
        pid = (row.get("participant_id") or "").strip()
        if not pid:
            continue
        rid = row.get("record_id") or ""
        if rid:
            para_by_pid[pid].add(rid)
        meta.setdefault(pid, {"participant_id": pid, "subcorpus": row.get("subcorpus", "")})

        src = row.get("label_source") or ""
        if _is_pv_label(src):
            pv_rows[pid] += 1
            code = (row.get("pv_code") or "").strip()
            if code:
                pv_cnt[pid][code] += 1
                subcode = (row.get("pv_subcode") or "").strip()
                pv_sub_cnt[pid][f"{code}|{subcode}"] += 1
        elif _is_gap_label(src):
            gap_rows[pid] += 1
            code = (row.get("theme_code") or "").strip()
            if code:
                ash_cnt[pid][code] += 1
        elif (row.get("theme_code") or "").strip():
            gap_rows[pid] += 1
            ash_cnt[pid][(row.get("theme_code") or "").strip()] += 1

    all_pids = sorted(meta.keys())
    pv_codes = sorted({c for ctr in pv_cnt.values() for c in ctr})
    ash_codes = sorted({c for ctr in ash_cnt.values() for c in ctr})

    out: list[dict] = []
    for pid in all_pids:
        n_para = max(len(para_by_pid[pid]), 1)
        row: dict = dict(meta[pid])
        row["n_paragraphs"] = len(para_by_pid[pid])

        pv_unique = 0
        gap_unique = 0
        for code in pv_codes:
            cnt = pv_cnt[pid].get(code, 0)
            dens = cnt / n_para
            row[f"pv_{code}_bin"] = 1 if cnt > 0 else 0
            row[f"pv_{code}_cnt"] = cnt
            row[f"pv_{code}_dens"] = round(dens, 4) if float_fmt else dens
            if cnt > 0:
                pv_unique += 1

        for code in ash_codes:
            cnt = ash_cnt[pid].get(code, 0)
            dens = cnt / n_para
            row[f"ash_{code}_bin"] = 1 if cnt > 0 else 0
            row[f"ash_{code}_cnt"] = cnt
            row[f"ash_{code}_dens"] = round(dens, 4) if float_fmt else dens
            if cnt > 0:
                gap_unique += 1

        total_labels = pv_rows[pid] + gap_rows[pid]
        row["pv_label_rows"] = pv_rows[pid]
        row["gap_label_rows"] = gap_rows[pid]
        row["total_label_rows"] = total_labels
        row["pv_burden_unique"] = pv_unique
        row["gap_burden_unique"] = gap_unique
        row["gap_ratio"] = round(gap_rows[pid] / total_labels, 4) if total_labels else 0.0

        for gname, codes in NMF_GROUP_CODES.items():
            cnt = sum(ash_cnt[pid].get(c, 0) for c in codes)
            row[f"dens_ash_{gname}"] = round(cnt / n_para, 4)

        for gname, codes in PV_GROUP_CODES.items():
            cnt = sum(pv_cnt[pid].get(c, 0) for c in codes)
            row[f"dens_pv_{gname}"] = round(cnt / n_para, 4)

        # Fine-grained PV (Code, Sub-code) densities so SDOH themes are distinct.
        for gname, combos in PV_SUBCODE_GROUPS.items():
            cnt = 0
            for c, s in combos:
                if s == "*":
                    cnt += sum(
                        n for key, n in pv_sub_cnt[pid].items()
                        if key.split("|", 1)[0] == c
                    )
                else:
                    cnt += pv_sub_cnt[pid].get(f"{c}|{s}", 0)
            row[f"dens_pvsub_{gname}"] = round(cnt / n_para, 4)

        out.append(row)

    return out


def feature_columns_for_set(
    row: dict,
    feature_set: str,
    *,
    value_suffix: str = "dens",
    gap_exclude: tuple[str, ...] = (),
    feature_pool: str = "default",
) -> list[str]:
    """Select columns for pv_only | gap_only | combined (default: density features).

    feature_pool:
      - default: theme densities + burden meta (pv/gap label counts, burden_unique, gap_ratio)
      - theme_densities: theme/code densities only — matches original README step 11/15 pools
        (Partnership/SocioEmotional PV + NMF theme gaps; no burden meta dominating LOOCV)
    """
    if feature_pool not in ("default", "theme_densities"):
        raise ValueError(f"feature_pool must be default|theme_densities, got {feature_pool!r}")
    meta = {"participant_id", "subcorpus", "n_paragraphs", "cohort"}

    def _excluded_gap_col(k: str) -> bool:
        if not gap_exclude:
            return False
        # Per-theme columns: ash_<code>_<suffix> (e.g. ash_mental_distress_dens)
        for code in gap_exclude:
            if k.startswith(f"ash_{code}_"):
                return True
        # Bundled group densities: dens_ash_<group>. A group column sums several
        # theme codes (see NMF_GROUP_CODES). If ANY member code is excluded the
        # column is contaminated and must be dropped to honor the non-redundant /
        # circularity-reduced set. NOTE: the old guard `k.startswith("ash_")`
        # short-circuited here and silently let dens_ash_* leak through.
        if k.startswith("dens_ash_"):
            gname = k[len("dens_ash_"):]
            members = NMF_GROUP_CODES.get(gname, (gname,))
            if any(m in gap_exclude for m in members):
                return True
        return False

    def _include_col(k: str, *, allow_meta: bool) -> bool:
        if k in meta:
            return False
        if feature_pool == "theme_densities" and k in BURDEN_META_COLS:
            return False
        if not allow_meta and k in BURDEN_META_COLS:
            return False
        return True

    pv_meta = () if feature_pool == "theme_densities" else ("pv_label_rows", "pv_burden_unique")
    gap_meta = () if feature_pool == "theme_densities" else (
        "gap_label_rows",
        "gap_burden_unique",
        "gap_ratio",
    )
    pv_cols = [
        k
        for k in row
        if _include_col(k, allow_meta=True)
        and (
            (k.startswith("pv_") and k.endswith(f"_{value_suffix}"))
            or k.startswith("dens_pv_")
            or k.startswith("dens_pvsub_")
            or k in pv_meta
        )
    ]
    gap_cols = [
        k
        for k in row
        if _include_col(k, allow_meta=True)
        and not _excluded_gap_col(k)
        and (
            (k.startswith("ash_") and k.endswith(f"_{value_suffix}"))
            or k.startswith("dens_ash_")
            or k in gap_meta
        )
    ]

    if feature_set == "pv_only":
        return sorted(pv_cols)
    if feature_set == "gap_only":
        return sorted(gap_cols)
    if feature_set == "combined":
        return sorted(set(pv_cols) | set(gap_cols))
    raise ValueError(f"feature_set must be pv_only|gap_only|combined, got {feature_set!r}")
