# app/modules/timeline.py

from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

from app.modules.identity_resolution import resolve_display_name

NEGATIVE_EMOTIONS = {"sad"}  # facial is 3-class


def _most_common(items: List[str]) -> Optional[str]:
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def detect_mood_changes(
    facial_sentiment: Dict[str, Any],
    text_sentiment: Dict[str, Any],
    speaker_segments: List[Dict[str, Any]],
    speaker_labeled_segments: Optional[List[Dict[str, Any]]] = None,
    speaker_to_person: Optional[Dict[str, str]] = None,
    person_to_name: Optional[Dict[str, str]] = None,
    *,
    alias_to_person: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    events = facial_sentiment.get("emotions_timeline", []) or []
    if not events:
        return []

    text_segments = (text_sentiment or {}).get("by_segment", []) or []

    speaker_labeled_segments = speaker_labeled_segments or []
    speaker_to_person = speaker_to_person or {}
    person_to_name = person_to_name or {}
    alias_to_person = alias_to_person or {}

    person_to_speakers = defaultdict(list)
    for sid, pid in (speaker_to_person or {}).items():
        person_to_speakers[pid].append(sid)

    def _seg_speaker_id(seg: Dict[str, Any]) -> Optional[str]:
        return seg.get("speaker_id") or seg.get("speaker")

    def canonical_person_id(pid: Optional[str]) -> Optional[str]:
        if not pid:
            return None
        if isinstance(pid, str) and pid.startswith("PERSON_"):
            return pid
        return alias_to_person.get(pid) or pid

    def find_context_for(pid: str, ts: float):
        speaker_ids = person_to_speakers.get(pid, [])
        best_seg = None
        best_overlap = 0.0
        best_sid = None

        for seg in speaker_labeled_segments:
            sid = _seg_speaker_id(seg)
            if speaker_ids and sid not in speaker_ids:
                continue

            s_start = float(seg.get("start", 0.0) or 0.0)
            s_end = float(seg.get("end", s_start) or s_start)
            if s_end <= s_start:
                continue

            overlap = (s_end - s_start) if (s_start <= ts <= s_end) else 0.0
            if overlap > best_overlap:
                best_overlap = overlap
                best_seg = seg
                best_sid = sid

        snippet = None
        if best_seg:
            txt = (best_seg.get("text") or "").strip()
            snippet = txt if len(txt) <= 140 else txt[:137].rstrip() + "..."

        best_text_sent = None
        best_gap = None
        for seg in text_segments:
            s_start = float(seg.get("start", 0.0) or 0.0)
            s_end = float(seg.get("end", s_start) or s_start)
            if s_end <= s_start:
                continue
            mid = 0.5 * (s_start + s_end)
            gap = abs(mid - ts)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_text_sent = seg

        return best_sid, snippet, best_text_sent

    per_person = defaultdict(list)
    for e in events:
        raw_pid = e.get("person_id")
        pid = canonical_person_id(raw_pid)
        if not pid:
            continue
        ts = float(e.get("timestamp", 0.0) or 0.0)
        emo = (e.get("emotion") or "neutral").lower()
        per_person[pid].append({"timestamp": ts, "emotion": emo})

    for pid in per_person:
        per_person[pid].sort(key=lambda x: x["timestamp"])

    mood_changes: List[Dict[str, Any]] = []

    WINDOW_SIZE = 5
    MIN_GAP_SEC = 10.0
    MIN_RUN = 2

    for pid, seq in per_person.items():
        if not seq:
            continue

        history: List[str] = []
        prev_state = "neutral"
        last_change_ts: Optional[float] = None

        run_state = prev_state
        run_len = 0

        for e in seq:
            ts = e["timestamp"]
            emo = e["emotion"]

            history.append(emo)
            if len(history) > WINDOW_SIZE:
                history.pop(0)

            smoothed = _most_common(history) or emo

            if smoothed == run_state:
                run_len += 1
            else:
                run_state = smoothed
                run_len = 1

            if smoothed == prev_state:
                continue

            if smoothed == "neutral":
                prev_state = smoothed
                continue

            if run_len < MIN_RUN:
                continue

            if last_change_ts is not None and (ts - last_change_ts) < MIN_GAP_SEC:
                prev_state = smoothed
                continue

            sid, snippet, text_seg = find_context_for(pid, ts)

            display_name = resolve_display_name(
                speaker_id=sid,
                person_id=pid,
                speaker_to_person=speaker_to_person,
                person_to_name=person_to_name,
            )
            if not display_name:
                display_name = sid or pid or "UNKNOWN"

            if smoothed in NEGATIVE_EMOTIONS:
                base_impact = 0.62
            elif smoothed == "happy":
                base_impact = 0.72
            else:
                base_impact = 0.60

            if snippet:
                base_impact += 0.04

            if smoothed in NEGATIVE_EMOTIONS and text_seg and isinstance(text_seg.get("sentiment"), dict):
                lbl = (text_seg["sentiment"].get("label") or "").lower()
                if lbl not in ("negative", "neg"):
                    base_impact *= 0.85

            change: Dict[str, Any] = {
                "timestamp": round(ts, 2),
                "person_id": pid,
                "speaker_id": sid,
                "display_name": display_name,
                "from": prev_state,
                "to": smoothed,
                "context": snippet,
                "impact_score": round(float(base_impact), 3),
            }

            if text_seg and isinstance(text_seg.get("sentiment"), dict):
                change["text_sentiment"] = {
                    "label": text_seg["sentiment"].get("label"),
                    "scores": text_seg["sentiment"].get("scores", {}),
                }

            mood_changes.append(change)
            prev_state = smoothed
            last_change_ts = ts

    mood_changes.sort(key=lambda x: x["timestamp"])
    return mood_changes


def create_timeline(
    speaker_segments: List[Dict[str, Any]],
    mood_changes: List[Dict[str, Any]],
    face_tracking_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    timeline: List[Dict[str, Any]] = []

    for seg in speaker_segments or []:
        speaker_id = seg.get("speaker_id") or seg.get("speaker")

        entry = {
            "type": "speech",
            "start": float(seg.get("start", 0.0) or 0.0),
            "end": float(seg.get("end", 0.0) or 0.0),
            "speaker": speaker_id,
        }

        if seg.get("speaker_name"):
            entry["speaker_name"] = seg.get("speaker_name")

        timeline.append(entry)

    for mc in mood_changes or []:
        entry = {
            "type": "mood_change",
            "timestamp": mc.get("timestamp"),
            "person_id": mc.get("person_id"),
            "speaker_id": mc.get("speaker_id"),
            "display_name": mc.get("display_name"),
            "from": mc.get("from"),
            "to": mc.get("to"),
            "context": mc.get("context"),
            "impact_score": mc.get("impact_score", 0.0),
        }
        if "text_sentiment" in mc:
            entry["text_sentiment"] = mc["text_sentiment"]
        timeline.append(entry)

    def _key(e: Dict[str, Any]) -> float:
        if e["type"] == "speech":
            return float(e.get("start", 0.0) or 0.0)
        return float(e.get("timestamp", 0.0) or 0.0)

    timeline.sort(key=_key)
    return timeline
