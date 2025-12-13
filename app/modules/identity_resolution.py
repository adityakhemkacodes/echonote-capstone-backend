# app/modules/identity_resolution.py
from typing import Dict, Optional, Any


def resolve_display_name(
    *,
    speaker_id: Optional[str] = None,
    person_id: Optional[str] = None,
    speaker_to_person: Optional[Dict[str, str]] = None,
    person_to_name: Optional[Dict[str, str]] = None,
    # NEW (optional): map any "face person id" (or alt id) -> canonical PERSON_k
    alias_to_person: Optional[Dict[str, str]] = None,
    fallback_speaker_prefix: str = "Speaker",
    fallback_person_prefix: str = "Person",
) -> str:
    """
    Resolve a human display name from speaker_id/person_id using available mappings.

    Priority:
      1) person_id -> name in person_to_name
      2) alias_to_person[person_id] -> name   (NEW, helps when facial ids != PERSON_k)
      3) speaker_id -> person_id (speaker_to_person) -> name
      4) speaker_id key directly in person_to_name (defensive)
      5) fallback pretty labels
    """
    speaker_to_person = speaker_to_person or {}
    person_to_name = person_to_name or {}
    alias_to_person = alias_to_person or {}

    # 1) Direct person_id -> name
    if person_id:
        nm = person_to_name.get(person_id)
        if isinstance(nm, str) and nm.strip():
            return nm.strip()

    # 2) person_id is an alias (e.g., face id) -> canonical person -> name
    if person_id:
        canonical_pid = alias_to_person.get(person_id)
        if canonical_pid:
            nm = person_to_name.get(canonical_pid)
            if isinstance(nm, str) and nm.strip():
                return nm.strip()

    # 3) speaker_id -> person_id -> name
    if speaker_id:
        pid = speaker_to_person.get(speaker_id)
        if pid:
            nm = person_to_name.get(pid)
            if isinstance(nm, str) and nm.strip():
                return nm.strip()

    # 4) Defensive: speaker_id keyed directly in person_to_name
    if speaker_id:
        nm = person_to_name.get(speaker_id)
        if isinstance(nm, str) and nm.strip():
            return nm.strip()

    # 5) Fallback labels
    if speaker_id:
        s = str(speaker_id).strip()
        if s.startswith("SPEAKER_"):
            s = s[len("SPEAKER_") :]
        s = s.replace("_", " ").strip()
        if s:
            return f"{fallback_speaker_prefix}: {s}"

    if person_id:
        p = str(person_id).strip()
        if p.startswith("PERSON_"):
            p = p[len("PERSON_") :]
        p = p.replace("_", " ").strip()
        if p:
            return f"{fallback_person_prefix}: {p}"

    return "Unknown"
