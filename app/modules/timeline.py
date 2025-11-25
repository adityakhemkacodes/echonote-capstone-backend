# app/modules/timeline.py

from typing import List, Dict, Any, Optional


def _calculate_impact(from_emotion: str, to_emotion: str) -> float:
    """
    Calculate impact score of an emotion change.

    Higher score = more significant change.
    """
    # Define emotion valence (you can tweak this set)
    positive_emotions = {"happy", "surprise"}
    negative_emotions = {"sad", "angry", "fear", "disgust"}

    from_positive = from_emotion in positive_emotions
    to_positive = to_emotion in positive_emotions

    # Big swing from positive to negative
    if from_positive and not to_positive:
        return 0.9
    # Big swing from negative to positive
    elif not from_positive and to_positive:
        return 0.8
    # Neutral ↔ non-neutral or same-valence changes
    else:
        return 0.5


def _filter_facial_mood_changes(
    changes: List[Dict[str, Any]], min_gap: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Remove instant flip-flop mood changes (e.g., neutral→fear→neutral)
    at the same or very close timestamps.

    Assumes `changes` are sorted by timestamp.
    """
    if not changes:
        return []

    filtered: List[Dict[str, Any]] = []
    last_kept: Optional[Dict[str, Any]] = None

    for ch in changes:
        if last_kept is None:
            filtered.append(ch)
            last_kept = ch
            continue

        same_time = abs(ch["timestamp"] - last_kept["timestamp"]) < 1e-6
        reversed_pair = (
            ch.get("from_emotion") == last_kept.get("to_emotion")
            and ch.get("to_emotion") == last_kept.get("from_emotion")
        )
        too_close = ch["timestamp"] - last_kept["timestamp"] < min_gap

        # Case 1: exact flip-flop at same timestamp -> ignore this one
        if same_time and reversed_pair:
            continue

        # Case 2: flip back very quickly -> treat as noise
        if reversed_pair and too_close:
            continue

        filtered.append(ch)
        last_kept = ch

    return filtered


def _find_speaker_for_time(
    speaker_segments: List[Dict[str, Any]], t: float
) -> Optional[str]:
    """
    Given diarization segments and a timestamp t, find which speaker
    was talking at that moment (if any).
    """
    best_speaker = None
    best_overlap = 0.0

    for seg in speaker_segments or []:
        try:
            s_start = float(seg.get("start", 0.0) or 0.0)
            s_end = float(seg.get("end", s_start) or s_start)
        except Exception:
            continue

        if s_end <= s_start:
            continue

        # Simple inclusion check + tiny overlap window
        if s_start <= t <= s_end:
            overlap = s_end - s_start
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg.get("speaker", None)

    return best_speaker


def detect_mood_changes(
    facial_sentiment: Dict[str, Any],
    text_sentiment: Dict[str, Any],
    speaker_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Detect meaningful mood changes primarily from facial emotion timeline.

    Steps:
      1) scan facial emotion timeline in order
      2) detect changes in emotion
      3) remove flip-flop / very short-gap noise
      4) attach impact scores
      5) (optional) attach which speaker was active at that time

    Returns a list of dicts like:
      {
        "timestamp": float,
        "from_emotion": str,
        "to_emotion": str,
        "person_id": str | None,
        "speaker_id": str | None,
        "impact_score": float,
      }
    """
    emo_timeline = facial_sentiment.get("emotions_timeline", []) or []
    if not emo_timeline:
        return []

    # Sort by timestamp for consistent processing
    emo_timeline = sorted(emo_timeline, key=lambda x: x.get("timestamp", 0.0))

    changes: List[Dict[str, Any]] = []

    # Initialize with the earliest emotion
    last_emotion = emo_timeline[0].get("emotion", "neutral")
    last_ts = float(emo_timeline[0].get("timestamp", 0.0) or 0.0)
    last_person = emo_timeline[0].get("person_id")

    for entry in emo_timeline[1:]:
        ts = float(entry.get("timestamp", 0.0) or 0.0)
        emotion = entry.get("emotion", "neutral")
        person_id = entry.get("person_id")

        if emotion != last_emotion:
            change = {
                "timestamp": ts,
                "from_emotion": last_emotion,
                "to_emotion": emotion,
                "person_id": person_id or last_person,
                "raw": True,  # mark as raw before filtering
            }
            changes.append(change)

            last_emotion = emotion
            last_ts = ts
            last_person = person_id or last_person

    # Filter flip-flop / noisy changes
    changes = sorted(changes, key=lambda x: x["timestamp"])
    changes = _filter_facial_mood_changes(changes, min_gap=1.0)

    # Attach impact score and speaker_id (via diarization)
    for ch in changes:
        ch["impact_score"] = _calculate_impact(
            ch["from_emotion"], ch["to_emotion"]
        )

        # Map to active speaker at that timestamp if diarization is available
        speaker_id = _find_speaker_for_time(speaker_segments, ch["timestamp"])
        if speaker_id is not None:
            ch["speaker_id"] = speaker_id

        # Raw flag no longer useful outside
        ch.pop("raw", None)

    # Sort by impact score descending (most important first)
    changes = sorted(changes, key=lambda x: x["impact_score"], reverse=True)

    return changes


def create_timeline(
    speaker_segments: List[Dict[str, Any]],
    mood_changes: List[Dict[str, Any]],
    face_tracks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build a unified chronological timeline of:
      - speaker events
      - mood change events
      - face appearance events

    The output is a list of events like:
      {
        "timestamp": float,
        "type": "speech" | "mood_change" | "face",
        ...additional fields...
      }
    """
    timeline: List[Dict[str, Any]] = []

    # 1) Speaker events from diarization
    for seg in speaker_segments or []:
        try:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
        except Exception:
            continue

        if end <= start:
            continue

        timeline.append(
            {
                "timestamp": start,
                "type": "speech",
                "speaker": seg.get("speaker", "UNKNOWN"),
                "end": end,
            }
        )

    # 2) Mood change events
    for mc in mood_changes or []:
        try:
            ts = float(mc.get("timestamp", 0.0) or 0.0)
        except Exception:
            continue

        event = {
            "timestamp": ts,
            "type": "mood_change",
            "from": mc.get("from_emotion", "neutral"),
            "to": mc.get("to_emotion", "neutral"),
            "impact_score": mc.get("impact_score", 0.0),
        }

        # Propagate identity if present
        if "person_id" in mc:
            event["person_id"] = mc["person_id"]
        if "speaker_id" in mc:
            event["speaker_id"] = mc["speaker_id"]

        timeline.append(event)

    # 3) Face appearance events (lightweight)
    for f in (face_tracks or [])[:100]:
        try:
            ts = float(f.get("timestamp", 0.0) or 0.0)
        except Exception:
            continue

        timeline.append(
            {
                "timestamp": ts,
                "type": "face",
                "person_id": f.get("person_id", "Unknown"),
            }
        )

    # Final sorting by time
    timeline.sort(key=lambda x: x["timestamp"])
    return timeline
