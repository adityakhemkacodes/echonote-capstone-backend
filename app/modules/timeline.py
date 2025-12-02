# app/modules/timeline.py

from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}


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
) -> List[Dict[str, Any]]:
    """
    Detect meaningful mood changes from the facial emotion timeline, with:
      - per-person smoothing
      - ignoring spurious negative states (rare 'sad/angry' blips)
      - minimum time gap between reported changes
      - optional context snippet & speaker name
      - optional text sentiment context around the same time

    Returns a list of:
      {
        "timestamp": float,
        "person_id": "PERSON_1",
        "speaker_id": "SPEAKER_00" | None,
        "speaker_name": "Aditya Khemka" | None,
        "from": "neutral",
        "to": "happy",
        "context": "short text snippet around that time" | None,
        "impact_score": float,
        "text_sentiment": {
          "label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
          "scores": { "positive": float, "negative": float, "neutral": float }
        } | None,
      }
    """

    events = facial_sentiment.get("emotions_timeline", []) or []
    if not events:
        return []

    # Text sentiment segments from analyze_text_sentiment
    text_segments = (text_sentiment or {}).get("by_segment", []) or []

    # Group facial events per person, sorted by time
    per_person = defaultdict(list)
    for e in events:
        pid = e.get("person_id")
        ts = float(e.get("timestamp", 0.0) or 0.0)
        emo = (e.get("emotion") or "neutral").lower()
        if not pid:
            continue
        per_person[pid].append({"timestamp": ts, "emotion": emo})

    for pid in per_person:
        per_person[pid].sort(key=lambda x: x["timestamp"])

    # Pre-compute emotion frequency per person to detect rare negative states
    per_person_counts: Dict[str, Counter] = {}
    for pid, seq in per_person.items():
        counts = Counter(e["emotion"] for e in seq)
        per_person_counts[pid] = counts

    mood_changes: List[Dict[str, Any]] = []

    # Helper: mapping + context
    speaker_labeled_segments = speaker_labeled_segments or []
    speaker_to_person = speaker_to_person or {}
    person_to_name = person_to_name or {}

    # Reverse mapping: person_id -> list of speaker_ids
    person_to_speakers = defaultdict(list)
    for sid, pid in speaker_to_person.items():
        person_to_speakers[pid].append(sid)

    def find_context_for(pid: str, ts: float):
        """
        Return:
          - speaker_id
          - speaker_name
          - transcript snippet around ts for this person
          - nearest text sentiment segment (for label/scores)
        """
        speaker_ids = person_to_speakers.get(pid, [])
        best_seg = None
        best_overlap = 0.0
        best_sid = None

        # 1) Find best diarized segment for snippet + speaker_id
        for seg in speaker_labeled_segments:
            sid = seg.get("speaker_id")
            if speaker_ids and sid not in speaker_ids:
                continue

            s_start = float(seg.get("start", 0.0) or 0.0)
            s_end = float(seg.get("end", s_start) or s_start)
            if s_end <= s_start:
                continue

            # Overlap or closeness between ts and [s_start, s_end]
            if s_start <= ts <= s_end:
                overlap = s_end - s_start
            else:
                overlap = max(0.0, min(ts, s_end) - max(ts, s_start))

            if overlap > best_overlap:
                best_overlap = overlap
                best_seg = seg
                best_sid = sid

        snippet = None
        if best_seg:
            text = (best_seg.get("text") or "").strip()
            snippet = text if len(text) <= 140 else text[:137].rstrip() + "..."

        speaker_name = None
        if best_sid and best_sid in speaker_to_person:
            p = speaker_to_person[best_sid]
            speaker_name = person_to_name.get(p)

        # 2) Attach nearest text sentiment segment in time
        best_text_sent = None
        best_ts_gap = None
        for seg in text_segments:
            s_start = float(seg.get("start", 0.0) or 0.0)
            s_end = float(seg.get("end", s_start) or s_start)
            if s_end <= s_start:
                continue
            mid = 0.5 * (s_start + s_end)
            gap = abs(mid - ts)
            if best_ts_gap is None or gap < best_ts_gap:
                best_ts_gap = gap
                best_text_sent = seg

        return best_sid, speaker_name, snippet, best_text_sent

    # Parameters to control sensitivity
    WINDOW_SIZE = 3          # number of recent states for smoothing
    MIN_GAP_SEC = 6.0        # minimum seconds between reported changes per person
    NEG_SHARE_THRESHOLD = 0.10  # ignore negative emotions if they are < 10% of this person's frames

    for pid, seq in per_person.items():
        if not seq:
            continue

        counts = per_person_counts[pid]
        total = sum(counts.values()) or 1
        allowed_negative = {
            emo
            for emo, c in counts.items()
            if emo in NEGATIVE_EMOTIONS and (c / total) >= NEG_SHARE_THRESHOLD
        }

        history: List[str] = []  # recent raw emotions for smoothing
        prev_state = "neutral"
        last_change_ts: Optional[float] = None

        for e in seq:
            ts = e["timestamp"]
            emo = e["emotion"]

            # Map very rare negative emotions back to neutral (noise)
            if emo in NEGATIVE_EMOTIONS and emo not in allowed_negative:
                emo = "neutral"

            history.append(emo)
            if len(history) > WINDOW_SIZE:
                history.pop(0)

            smoothed = _most_common(history) or emo

            # Only consider changes away from previous smoothed state
            if smoothed == prev_state:
                continue

            # Ignore transitions TO neutral – focus on when mood becomes
            # noticeably non-neutral (or changes between non-neutrals)
            if smoothed == "neutral":
                prev_state = smoothed
                continue

            # Enforce minimum time gap between reported changes
            if last_change_ts is not None and (ts - last_change_ts) < MIN_GAP_SEC:
                prev_state = smoothed
                continue

            # At this point, we accept this as a "real" mood change
            sid, speaker_name, snippet, text_seg = find_context_for(pid, ts)

            # Crude impact score:
            #  - negative emotions a bit higher than positive ones
            #  - slight boost if we have a context snippet
            base_impact = 0.7
            if smoothed in NEGATIVE_EMOTIONS:
                base_impact = 0.9
            elif smoothed == "happy":
                base_impact = 0.8

            if snippet:
                base_impact += 0.05

            change: Dict[str, Any] = {
                "timestamp": round(ts, 2),
                "person_id": pid,
                "speaker_id": sid,
                "speaker_name": speaker_name,
                "from": prev_state,
                "to": smoothed,
                "context": snippet,
                "impact_score": round(base_impact, 3),
            }

            # Attach local text sentiment if available
            if text_seg and isinstance(text_seg.get("sentiment"), dict):
                change["text_sentiment"] = {
                    "label": text_seg["sentiment"].get("label"),
                    "scores": text_seg["sentiment"].get("scores", {}),
                }

            mood_changes.append(change)

            prev_state = smoothed
            last_change_ts = ts

    # Sort all changes by time
    mood_changes.sort(key=lambda x: x["timestamp"])
    return mood_changes


def create_timeline(
    speaker_segments: List[Dict[str, Any]],
    mood_changes: List[Dict[str, Any]],
    face_tracking_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build a simple merged timeline of:
      - speech segments
      - mood change events

    This is primarily for UI / debugging and is trimmed by the processor.
    """

    timeline: List[Dict[str, Any]] = []

    # 1) Speaker segments (who spoke when)
    for seg in speaker_segments or []:
        entry = {
            "type": "speech",
            "start": float(seg.get("start", 0.0) or 0.0),
            "end": float(seg.get("end", 0.0) or 0.0),
            "speaker": seg.get("speaker"),
        }
        timeline.append(entry)

    # 2) Mood changes as point events
    for mc in mood_changes or []:
        entry = {
            "type": "mood_change",
            "timestamp": mc.get("timestamp"),
            "person_id": mc.get("person_id"),
            "speaker_id": mc.get("speaker_id"),
            "speaker_name": mc.get("speaker_name"),
            "from": mc.get("from"),
            "to": mc.get("to"),
            "context": mc.get("context"),
            "impact_score": mc.get("impact_score", 0.0),
        }
        # If we stored local text sentiment, surface it on the timeline too
        if "text_sentiment" in mc:
            entry["text_sentiment"] = mc["text_sentiment"]
        timeline.append(entry)

    # Sort combined timeline:
    def _key(e: Dict[str, Any]) -> float:
        if e["type"] == "speech":
            return float(e.get("start", 0.0) or 0.0)
        return float(e.get("timestamp", 0.0) or 0.0)

    timeline.sort(key=_key)
    return timeline