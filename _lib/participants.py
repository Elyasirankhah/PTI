from __future__ import annotations

import re


def participant_id_from_source(source: str) -> str:
    """Extract stable participant ID from transcript path."""
    s = source.replace("\\", "/")
    m = re.search(r"/(P\d+)\b", s, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"/(SS\d+)\s", s, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"(P\d+)", s, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"(SS\d+)", s, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"(IDI[\w\-]+)", s, re.I)
    if m:
        return m.group(1)
    stem = s.rsplit("/", 1)[-1]
    stem = re.sub(r"\.docx$", "", stem, flags=re.I).strip()
    return stem or "unknown"


def cohort_from_source(source: str) -> str:
    s = source.replace("\\", "/")
    if "/Depressed/" in s:
        return "Depressed"
    if "/Non-Depressed/" in s:
        return "Non-Depressed"
    if "/TrialParticipants/" in s:
        return "TrialParticipants"
    if "Pakistan" in s or "DemoCorpusB" in s:
        return "DemoCorpusB"
    return "unknown"


def demo_depressed_outcome(cohort: str) -> str:
    if cohort == "Depressed":
        return "1"
    if cohort == "Non-Depressed":
        return "0"
    return ""
