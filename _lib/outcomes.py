from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from _lib.filters import is_participant_turn
from _lib.participants import cohort_from_source, demo_depressed_outcome, participant_id_from_source
from _lib.sites import DEMO_REGRESSION_COHORTS, DEMO_SUBCORPUS, DEMO_INDEPENDENT_DISTRESS_KEYWORDS

OUTCOME_FIELDS = [
    "participant_id",
    "subcorpus",
    "cohort",
    "include_in_regression",
    "include_in_elevated",
    "depressed",
    "elevated_stratum",
    "text_distress",
    "text_distress_score",
    "suicide_attempt",
    "suicide_ideation",
    "phq9_score",
    "phq9_ge9",
    "bdi_score",
    "bdi_ge10",
    "study_case",
    "mother_age",
    "child_age_months",
    "child_sex",
    "education",
    "ethnicity",
    "religion",
    "n_children",
    "household_size",
    "notes",
]

DEMO_CLINICAL_COPY_COLS = [
    "phq9_score",
    "bdi_score",
    "mother_age",
    "child_age_months",
    "child_sex",
    "education",
    "ethnicity",
    "religion",
    "n_children",
    "household_size",
    "study_case",
]


def _blob(rows: list[dict]) -> str:
    return " ".join((r.get("context") or "") for r in rows).lower()


def _participant_blob(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        if is_participant_turn(row, strict=True):
            parts.append((row.get("context") or "").lower())
    return " ".join(parts)


def _has_any(blob: str, phrases: tuple[str, ...]) -> bool:
    return any(p in blob for p in phrases)


def _distress_hits(blob: str) -> int:
    return sum(1 for p in DEMO_INDEPENDENT_DISTRESS_KEYWORDS if p in blob)


def _demo_elevated_stratum(cohort: str) -> str:
    if cohort == "Non-Depressed":
        return "0"
    if cohort in ("Depressed", "TrialParticipants"):
        return "1"
    return ""


def infer_outcomes_from_units(units: list[dict]) -> list[dict]:
    """Build one outcome row per participant from full corpus (whole interview, all paragraphs)."""
    by_pid: dict[str, list[dict]] = defaultdict(list)
    meta: dict[str, dict] = {}

    for row in units:
        if (row.get("TO_PAT_YN") or "N").upper() != "N":
            continue
        src = row.get("source") or ""
        pid = participant_id_from_source(src)
        by_pid[pid].append(row)
        if pid not in meta:
            cohort = cohort_from_source(src)
            sub = row.get("subcorpus") or ""
            include = "1"
            include_elevated = "1"
            depressed = demo_depressed_outcome(cohort)
            elevated = _demo_elevated_stratum(cohort)
            suicide_attempt = ""
            suicide_ideation = ""
            notes = ""

            if sub == DEMO_SUBCORPUS:
                if cohort not in DEMO_REGRESSION_COHORTS:
                    include = "0"
                    notes = "TrialParticipants excluded from legacy depressed regression"
            elif sub == DEMO_SUBCORPUS_B:
                depressed = ""
                elevated = ""
                include_elevated = "1"
                suicide_ideation = "1"

            meta[pid] = {
                "participant_id": pid,
                "subcorpus": sub,
                "cohort": cohort,
                "include_in_regression": include,
                "include_in_elevated": include_elevated,
                "depressed": depressed,
                "elevated_stratum": elevated,
                "text_distress": "",
                "text_distress_score": "",
                "suicide_attempt": suicide_attempt,
                "suicide_ideation": suicide_ideation,
                "phq9_score": "",
                "phq9_ge9": "",
                "bdi_score": "",
                "bdi_ge10": "",
                "study_case": "",
                "mother_age": "",
                "child_age_months": "",
                "child_sex": "",
                "education": "",
                "ethnicity": "",
                "religion": "",
                "n_children": "",
                "household_size": "",
                "notes": notes,
            }

    for pid, rows in by_pid.items():
        if meta[pid]["subcorpus"] == DEMO_SUBCORPUS_B:
            blob = _blob(rows)
            meta[pid]["suicide_attempt"] = "1" if _has_any(blob, PAK_SUICIDE_ATTEMPT_KEYWORDS) else "0"
            if _has_any(blob, PAK_SUICIDE_IDEATION_KEYWORDS):
                meta[pid]["suicide_ideation"] = "1"
            continue

        if meta[pid]["subcorpus"] != DEMO_SUBCORPUS:
            continue

        p_blob = _participant_blob(rows)
        hits = _distress_hits(p_blob)
        meta[pid]["text_distress_score"] = str(hits)
        meta[pid]["text_distress"] = "1" if hits > 0 else "0"

    return [meta[k] for k in sorted(meta.keys())]


def load_outcomes(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_regression_pids(
    outcomes_path: Path,
    *,
    require_depressed: bool = True,
    subcorpus: str | None = None,
) -> set[str]:
    """Regression cohort PIDs for one subcorpus (default: DemoCorpus)."""
    sub_filter = subcorpus or DEMO_SUBCORPUS
    pids: set[str] = set()
    for row in load_outcomes(outcomes_path):
        if row.get("subcorpus") != sub_filter:
            continue
        if (row.get("include_in_regression") or "").strip() != "1":
            continue
        if require_depressed and (row.get("depressed") or "").strip() not in {"0", "1"}:
            continue
        pids.add(row["participant_id"].strip().upper())
    return pids


def write_outcomes(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTCOME_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def merge_manual_overrides(auto_rows: list[dict], override_path: Path | None) -> list[dict]:
    if override_path is None or not override_path.is_file():
        return auto_rows
    overrides = {
        r["participant_id"]: r
        for r in load_outcomes(override_path)
        if not (r.get("participant_id") or "").startswith("#")
    }
    merged = []
    for row in auto_rows:
        pid = row["participant_id"]
        if pid not in overrides:
            merged.append(row)
            continue
        o = overrides[pid]
        new_row = dict(row)
        for k in OUTCOME_FIELDS:
            val = (o.get(k) or "").strip()
            if val and not val.startswith("#"):
                new_row[k] = val
        merged.append(new_row)
    return merged
