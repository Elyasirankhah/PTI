from __future__ import annotations

import re

# Explicit interviewer turn markers (DOCX transcriptions use I:, I;, I;;, Q:, etc.)
_INTERVIEWER_PREFIX = re.compile(
    r"^(Interviewer|INT|Q|I)\s*[:;]+\s*",
    re.I,
)

# Interviewer field notes / reflection — not participant voice (even if TO_PAT_YN=N)
_JUNK_PATTERNS = [
    re.compile(r"^reflect and think about the information", re.I),
    re.compile(r"^reflect on your role as an interviewer", re.I),
    re.compile(r"when i reached at the respondent", re.I),
    re.compile(r"appropriate environment to conduct interview", re.I),
    re.compile(r"while conducting kii with participant", re.I),
    re.compile(r"challenges while doing interview", re.I),
    re.compile(r"^field notes\b", re.I),
    re.compile(r"^write a short description of the participant", re.I),
    re.compile(r"what information do you wish", re.I),
    re.compile(r"^i brought her in the health post with me", re.I),
    re.compile(r"^there were not many challenges while doing interview", re.I),
    re.compile(r"^it was about \d+.*when i reached", re.I),
    re.compile(r"^while talking about the participant she spoke", re.I),
    re.compile(r"^during the whole interview, participant was scared", re.I),
    re.compile(r"^participant easily allocate the time", re.I),
    re.compile(r"^participant spoke openly during this interview", re.I),
    re.compile(r"^i got enough information from this interview", re.I),
    re.compile(r"^i think i could get more information", re.I),
    re.compile(r"^at first participant was scared because of her mother in-law", re.I),
    re.compile(r"^in our first meet, she spoke openly when i asked", re.I),
    re.compile(r"^when i went in her home", re.I),
    re.compile(r"^while conducting kii with participant", re.I),
]


def has_interviewer_prefix(text: str) -> bool:
    return bool(_INTERVIEWER_PREFIX.match((text or "").lstrip()))


def is_junk_paragraph(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 12:
        return True
    for pat in _JUNK_PATTERNS:
        if pat.search(t):
            return True
    return False


def is_participant_turn(row: dict, *, strict: bool = True) -> bool:
    """
    Participant voice only.
    strict=True: require turn_role=participant OR explicit P:/R:/Participant prefix.
    """
    text = (row.get("context") or "").lstrip()
    if is_junk_paragraph(text):
        return False
    if has_interviewer_prefix(text):
        return False
    if (row.get("TO_PAT_YN") or "N").strip().upper() != "N":
        return False
    role = (row.get("turn_role") or "").strip().lower()
    if role == "interviewer":
        return False
    has_prefix = bool(
        re.match(r"^(P|Participant|Respondent|R)\s*[:.\-–—]\s*", text, re.I)
    )
    if strict:
        return role == "participant" or has_prefix
    return role in ("participant", "unknown", "") or has_prefix


def filter_participant_rows(rows: list[dict], *, strict: bool = True) -> list[dict]:
    return [r for r in rows if is_participant_turn(r, strict=strict)]
