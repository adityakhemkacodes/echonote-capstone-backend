from collections import defaultdict, Counter
from typing import List, Dict, Optional
import difflib


def _normalize_name(s: str) -> str:
    return (s or "").strip().lower()


def _name_similarity(a: str, b: str) -> float:
    """
    Cheap fuzzy similarity between two name strings (0..1).
    Uses a mix of SequenceMatcher and token overlap.
    """
    a_norm = _normalize_name(a)
    b_norm = _normalize_name(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    if not a_tokens or not b_tokens:
        token_sim = 0.0
    else:
        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        token_sim = inter / union if union > 0 else 0.0

    char_sim = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()

    return 0.6 * char_sim + 0.4 * token_sim


def _cluster_names_local(names: List[str], threshold: float = 0.78) -> List[List[str]]:
    """
    Cluster name variants for a *single person* based on fuzzy similarity.
    """
    unique = [n for n in {n for n in names if n}]
    clusters: List[List[str]] = []

    for name in unique:
        placed = False
        for cluster in clusters:
            if _name_similarity(name, cluster[0]) >= threshold:
                cluster.append(name)
                placed = True
                break
        if not placed:
            clusters.append([name])

    return clusters


def _choose_canonical_name(aliases: List[str]) -> str:
    """
    Pick the nicest human-readable canonical name from aliases:
      - prefer 2+ tokens
      - then prefer longer
    """
    if not aliases:
        return ""

    cleaned = [(" ".join(a.split())).strip() for a in aliases if a and a.strip()]
    if not cleaned:
        return ""

    def score(n: str):
        tokens = n.split()
        return (min(len(tokens), 3), len(n))

    cleaned.sort(key=score, reverse=True)
    return cleaned[0]


def fuse_identities(
    face_tracks: List[Dict],
    ocr_names: List[Dict],
    speaker_segments: List[Dict],
    transcript_names: Optional[List[str]] = None,  # kept for future use / signature
):
    """
    SAFEST + CORRECT version:
      - OCR names are linked to person_id via timestamp
      - Name clustering is done *PER PERSON*, not globally
      - Transcript names are NOT used for identity assignment
    """

    # ---------------------------
    # STEP 1 — Assign OCR names → person_id
    # ---------------------------
    person_name_votes = defaultdict(list)

    for entry in ocr_names or []:
        ts = entry.get("timestamp")
        name = entry.get("name")
        if ts is None or not name:
            continue

        best_pid = None
        best_dt = 999.0

        for f in face_tracks or []:
            if "timestamp" not in f or "person_id" not in f:
                continue
            dt = abs(float(f["timestamp"]) - float(ts))
            if dt < best_dt:
                best_dt = dt
                best_pid = f["person_id"]

        if best_pid is not None:
            person_name_votes[best_pid].append(name)

    # ---------------------------
    # STEP 2 — speaker_id → person_id
    # ---------------------------
    speaker_votes = defaultdict(list)

    for seg in speaker_segments or []:
        sid = seg.get("speaker")
        start, end = seg.get("start"), seg.get("end")
        if sid is None or start is None or end is None:
            continue

        mid = (float(start) + float(end)) / 2.0

        best_pid, best_dt = None, 999.0
        for f in face_tracks or []:
            if "timestamp" not in f or "person_id" not in f:
                continue
            dt = abs(float(f["timestamp"]) - mid)
            if dt < best_dt:
                best_dt = dt
                best_pid = f["person_id"]

        if best_pid is not None:
            speaker_votes[sid].append(best_pid)

    speaker_to_person = {
        sid: Counter(pids).most_common(1)[0][0]
        for sid, pids in speaker_votes.items()
        if pids
    }

    # Invert mapping -> person_id → [speaker_ids]
    person_to_speakers = defaultdict(list)
    for sid, pid in speaker_to_person.items():
        person_to_speakers[pid].append(sid)

    # ---------------------------
    # STEP 3 — Build participants safely
    # ---------------------------
    all_pids = {
        f["person_id"] for f in (face_tracks or []) if "person_id" in f
    }
    participants = []
    person_to_name = {}

    for pid in sorted(all_pids):
        ocr_list = person_name_votes.get(pid, [])

        if not ocr_list:
            participants.append(
                {
                    "participant_id": pid,
                    "canonical_name": None,
                    "aliases": [],
                    "speaker_ids": person_to_speakers.get(pid, []),
                    "face_person_id": pid,
                }
            )
            continue

        # Cluster *per-person* OCR names
        clusters = _cluster_names_local(ocr_list)
        flat_aliases = [item for cluster in clusters for item in cluster]

        canonical = _choose_canonical_name(flat_aliases)

        participants.append(
            {
                "participant_id": pid,
                "canonical_name": canonical,
                "aliases": sorted(set(flat_aliases)),
                "speaker_ids": person_to_speakers.get(pid, []),
                "face_person_id": pid,
            }
        )

        if canonical:
            person_to_name[pid] = canonical

    return participants, speaker_to_person, person_to_name
