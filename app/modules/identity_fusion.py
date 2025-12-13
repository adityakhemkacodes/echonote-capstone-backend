# app/modules/identity_fusion.py
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple, Any
import difflib


def _normalize_name(s: str) -> str:
    return (s or "").strip().lower()


def _name_similarity(a: str, b: str) -> float:
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


# ============================================================
# Legacy path (OCR + diarization) — keep for fallback
# ============================================================

def fuse_identities(
    face_tracks: List[Dict],
    ocr_names: List[Dict],
    speaker_segments: List[Dict],
    transcript_names: Optional[List[str]] = None,
):
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

    person_to_speakers = defaultdict(list)
    for sid, pid in speaker_to_person.items():
        person_to_speakers[pid].append(sid)

    all_pids = {f["person_id"] for f in (face_tracks or []) if "person_id" in f}
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


# ============================================================
# NEW path (Zoom per-speaker audio) — primary going forward
# ============================================================

def fuse_identities_from_zoom(
    face_tracks: Optional[List[Dict[str, Any]]],
    speaker_registry: List[Dict[str, Any]],
    *,
    speaker_labeled_segments: Optional[List[Dict[str, Any]]] = None,
    auto_align_faces: bool = True,
    max_face_time_gap_sec: float = 1.25,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """
    Zoom-first identity fusion.

    Returns (legacy-compatible shapes):
      participants: [
        {
          "participant_id": "PERSON_0",
          "canonical_name": "Aditya Khemka",
          "aliases": ["Aditya Khemka"],
          "speaker_ids": ["SPEAKER_aditya_khemka"],
          "face_person_id": "PERSON_7" (best-effort) or None
        }, ...
      ]
      speaker_to_person: { "SPEAKER_aditya_khemka": "PERSON_0", ... }
      person_to_name: { "PERSON_0": "Aditya Khemka", ... }
    """

    # --- local imports (keep module-level minimal + avoids circular surprises)
    from typing import Any
    from collections import defaultdict, Counter
    import bisect

    # --- normalize speaker registry into ordered unique list
    speakers: List[Tuple[str, Optional[str]]] = []
    seen = set()
    for s in speaker_registry or []:
        if not isinstance(s, dict):
            continue
        sid = (s.get("speaker_id") or s.get("id") or "").strip()
        if not sid or sid.upper() == "UNKNOWN":
            continue
        if sid in seen:
            continue
        seen.add(sid)
        nm = (s.get("name") or s.get("display_name") or "").strip() or None
        speakers.append((sid, nm))

    # If no speakers, no participants
    if not speakers:
        return [], {}, {}

    # Build PERSON ids (stable ordering from registry)
    speaker_to_person: Dict[str, str] = {}
    person_to_name: Dict[str, str] = {}
    participants: List[Dict[str, Any]] = []

    for idx, (sid, nm) in enumerate(speakers):
        pid = f"PERSON_{idx}"
        speaker_to_person[sid] = pid
        if nm:
            person_to_name[pid] = nm

        participants.append(
            {
                "participant_id": pid,
                "canonical_name": nm,
                "aliases": [nm] if nm else [],
                "speaker_ids": [sid],
                "face_person_id": None,  # set later if aligned
            }
        )

    if not auto_align_faces:
        return participants, speaker_to_person, person_to_name

    # --- best-effort alignment: speaker -> face PERSON_k, using time proximity
    face_tracks = face_tracks or []
    speaker_labeled_segments = speaker_labeled_segments or []

    # Build sorted list of (timestamp, face_pid)
    face_points: List[Tuple[float, str]] = []
    for f in face_tracks:
        if not isinstance(f, dict):
            continue
        try:
            ts = float(f.get("timestamp", 0.0) or 0.0)
            fp = str(f.get("person_id") or "").strip()
            if fp:
                face_points.append((ts, fp))
        except Exception:
            continue

    face_points.sort(key=lambda x: x[0])

    # If we don't have both signals, we can't align
    if not face_points or not speaker_labeled_segments:
        return participants, speaker_to_person, person_to_name

    # Pre-split timestamps for binary search
    face_times = [t for (t, _) in face_points]

    def _seg_speaker_id(seg: Dict[str, Any]) -> str:
        return (seg.get("speaker_id") or seg.get("speaker") or "").strip()

    def nearest_face_pid(t: float) -> Optional[str]:
        """
        O(log N) nearest lookup using bisect.
        Returns pid if within max_face_time_gap_sec, else None.
        """
        if not face_times:
            return None

        i = bisect.bisect_left(face_times, t)

        best_pid = None
        best_dt = None

        # candidate: i
        if 0 <= i < len(face_points):
            dt = abs(face_points[i][0] - t)
            best_dt = dt
            best_pid = face_points[i][1]

        # candidate: i-1
        j = i - 1
        if 0 <= j < len(face_points):
            dt2 = abs(face_points[j][0] - t)
            if best_dt is None or dt2 < best_dt:
                best_dt = dt2
                best_pid = face_points[j][1]

        if best_dt is not None and best_dt <= float(max_face_time_gap_sec or 0.0):
            return best_pid
        return None

    # For each speaker utterance mid-time, vote nearest face pid
    votes: Dict[str, List[str]] = defaultdict(list)

    for seg in speaker_labeled_segments:
        if not isinstance(seg, dict):
            continue
        try:
            sid = _seg_speaker_id(seg)
            if not sid:
                continue

            s = float(seg.get("start", 0.0) or 0.0)
            e = float(seg.get("end", s) or s)
            if e < s:
                s, e = e, s

            mid = 0.5 * (s + e)
            fp = nearest_face_pid(mid)
            if fp:
                votes[sid].append(fp)
        except Exception:
            continue

    # Choose most common face pid per speaker, then propagate onto participant.face_person_id
    for p in participants:
        sids = p.get("speaker_ids") or []
        sid = sids[0] if sids else None
        if not sid:
            continue
        if sid not in votes or not votes[sid]:
            continue

        best_face_pid = Counter(votes[sid]).most_common(1)[0][0]
        p["face_person_id"] = best_face_pid

    return participants, speaker_to_person, person_to_name

