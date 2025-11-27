from typing import List, Dict, Any, Optional
from collections import defaultdict
import math

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _group_emotions_by_person(emotions_timeline: List[Dict[str, Any]]):
    """
    Group facial emotion samples by person_id.
    Each entry in emotions_timeline is expected to look like:
      { "timestamp": float, "emotion": "neutral", "person_id": "PERSON_1", ... }
    """
    by_person = defaultdict(list)
    for e in emotions_timeline or []:
        ts = float(e.get("timestamp", 0.0) or 0.0)
        emo = e.get("emotion") or "neutral"
        pid = e.get("person_id") or "UNKNOWN"
        by_person[pid].append((ts, emo))

    # sort each person's samples by time
    for pid in by_person:
        by_person[pid].sort(key=lambda x: x[0])

    return by_person


def _build_streaks(samples, min_duration: float):
    """
    Compress (timestamp, emotion) samples into streaks:
      [{ "start": .., "end": .., "emotion": "happy", "duration": .. }, ...]
    A streak is a run of the same emotion.
    Very short streaks (< min_duration) are kept but marked so we can
    ignore them when detecting mood changes.
    """
    if not samples:
        return []

    streaks = []
    cur_emo = samples[0][1]
    start_ts = samples[0][0]
    last_ts = start_ts

    for ts, emo in samples[1:]:
        if emo == cur_emo:
            last_ts = ts
            continue

        duration = max(0.01, last_ts - start_ts)
        streaks.append(
            {
                "start": start_ts,
                "end": last_ts,
                "emotion": cur_emo,
                "duration": duration,
                "is_short": duration < min_duration,
            }
        )

        # start new streak
        cur_emo = emo
        start_ts = ts
        last_ts = ts

    # final streak
    duration = max(0.01, last_ts - start_ts)
    streaks.append(
        {
            "start": start_ts,
            "end": last_ts,
            "emotion": cur_emo,
            "duration": duration,
            "is_short": duration < min_duration,
        }
    )

    return streaks


def _find_speaker_at_time(
    speaker_segments: List[Dict[str, Any]],
    t: float,
) -> Optional[str]:
    """
    Find diarized speaker id active at time t.
    """
    for seg in speaker_segments or []:
        s = float(seg.get("start", 0.0) or 0.0)
        e = float(seg.get("end", s) or s)
        if s <= t <= e:
            return seg.get("speaker")
    return None


def _extract_context_snippet(
    labeled_segments: List[Dict[str, Any]],
    t: float,
    window: float = 6.0,
    max_chars: int = 140,
) -> Dict[str, Any]:
    """
    Extract a small text snippet around time t from speaker-labeled
    transcript segments.

    Returns:
      { "snippet": str, "speaker_id": Optional[str] }
    """
    if not labeled_segments:
        return {"snippet": "", "speaker_id": None}

    # Collect segments that overlap [t - window, t + window]
    start_window = t - window
    end_window = t + window

    candidates = []
    for seg in labeled_segments:
        s = float(seg.get("start", 0.0) or 0.0)
        e = float(seg.get("end", s) or s)
        if e < start_window or s > end_window:
            continue
        candidates.append(seg)

    if not candidates:
        return {"snippet": "", "speaker_id": None}

    # Sort by time and concatenate text until we hit max_chars
    candidates.sort(key=lambda s: float(s.get("start", 0.0) or 0.0))
    full = []
    speaker_votes = defaultdict(int)

    for seg in candidates:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        full.append(text)
        speaker_id = seg.get("speaker_id")
        if speaker_id:
            speaker_votes[speaker_id] += 1

    if not full:
        return {"snippet": "", "speaker_id": None}

    joined = " ".join(full)
    if len(joined) > max_chars:
        joined = joined[: max_chars - 3].rstrip() + "..."

    # Pick the speaker that appears most in these segments
    speaker_id = None
    if speaker_votes:
        speaker_id = max(speaker_votes.items(), key=lambda x: x[1])[0]

    return {"snippet": joined, "speaker_id": speaker_id}


def _infer_context_type(snippet: str) -> Optional[str]:
    """
    Very simple heuristic to label the context:
      - 'joke / small talk'
      - 'issue / blocker'
      - 'decision / planning'
      - None
    """
    s = snippet.lower()

    if any(k in s for k in ["haha", "laugh", "joke", "lol", "pizza", "sushi", "coffee", "cat", "mascot"]):
        return "joke / small talk"

    if any(k in s for k in ["blocker", "problem", "issue", "error", "bug", "doesn't work", "locked", "crash"]):
        return "issue / blocker"

    if any(k in s for k in ["decided", "we decided", "we agreed", "decision", "choose", "go with", "plan", "roadmap"]):
        return "decision / planning"

    return None


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def detect_mood_changes(
    facial_sentiment: Dict[str, Any],
    text_sentiment: Dict[str, Any],
    speaker_segments: List[Dict[str, Any]],
    speaker_labeled_segments: Optional[List[Dict[str, Any]]] = None,
    speaker_to_person: Optional[Dict[str, str]] = None,
    min_streak_duration: float = 4.0,
    min_gap_seconds: float = 10.0,
    max_changes: int = 8,
) -> List[Dict[str, Any]]:
    """
    Detect *meaningful* mood changes from facial sentiment, with:
      - smoothing via emotion streaks
      - ignoring very short blips
      - enforcing a minimum time gap between reported changes
      - attaching person_id, speaker_id, and a short context snippet.

    Returns a list of dicts:
      {
        "timestamp": float,
        "from_emotion": "neutral",
        "to_emotion": "happy",
        "person_id": "PERSON_1",
        "speaker_id": "SPEAKER_00" | None,
        "context_snippet": "...",
        "context_type": "joke / small talk" | "issue / blocker" | ... | None,
        "impact_score": float,
      }
    """
    emotions_timeline = (facial_sentiment or {}).get("emotions_timeline", []) or []
    if not emotions_timeline:
        return []

    speaker_to_person = speaker_to_person or {}

    by_person = _group_emotions_by_person(emotions_timeline)
    changes: List[Dict[str, Any]] = []

    for pid, samples in by_person.items():
        streaks = _build_streaks(samples, min_streak_duration)
        if len(streaks) < 2:
            continue

        last_change_time = -1e9

        for i in range(1, len(streaks)):
            prev_s = streaks[i - 1]
            curr_s = streaks[i]

            if prev_s["emotion"] == curr_s["emotion"]:
                continue
            
            if curr_s["emotion"] == "neutral":
                continue

            # Require new emotion to be a reasonably long streak
            if curr_s["duration"] < min_streak_duration:
                continue

            change_time = curr_s["start"]

            # Enforce minimum gap between reported changes for this person
            if change_time - last_change_time < min_gap_seconds:
                continue

            last_change_time = change_time

            # Attach context snippet + speaker
            ctx = _extract_context_snippet(
                speaker_labeled_segments or [],
                change_time,
                window=6.0,
                max_chars=160,
            )
            snippet = ctx["snippet"]
            speaker_id = ctx["speaker_id"]

            # If diarization segments are also provided, fall back to them
            # to determine speaker when snippet didn't give us one.
            if speaker_id is None:
                speaker_id = _find_speaker_at_time(speaker_segments, change_time)

            # Try to infer a coarse context type (joke / blocker / decision)
            context_type = _infer_context_type(snippet) if snippet else None

            # Impact score ~ how long the new emotion lasts
            impact_score = curr_s["duration"]

            changes.append(
                {
                    "timestamp": round(change_time, 2),
                    "from_emotion": prev_s["emotion"],
                    "to_emotion": curr_s["emotion"],
                    "person_id": pid,
                    "speaker_id": speaker_id,
                    "context_snippet": snippet,
                    "context_type": context_type,
                    "impact_score": float(impact_score),
                }
            )

    # If we have many changes, keep the most impactful ones,
    # but sort final result by time for readability.
    if len(changes) > max_changes:
        changes.sort(key=lambda c: c["impact_score"], reverse=True)
        changes = changes[:max_changes]

    changes.sort(key=lambda c: c["timestamp"])
    return changes


def create_timeline(
    speaker_segments: List[Dict[str, Any]],
    mood_changes: List[Dict[str, Any]],
    face_tracking_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build a simple combined timeline of speaking segments and mood changes.

    Each entry is either:
      - { "type": "speech_segment", "start": .., "end": .., "speaker_id": .. }
      - { "type": "mood_change", ...same keys as detect_mood_changes(...) }
    """
    timeline: List[Dict[str, Any]] = []

    # Speech segments
    for seg in speaker_segments or []:
        s = float(seg.get("start", 0.0) or 0.0)
        e = float(seg.get("end", s) or s)
        timeline.append(
            {
                "type": "speech_segment",
                "start": s,
                "end": e,
                "speaker_id": seg.get("speaker"),
            }
        )

    # Mood changes (already have timestamp)
    for mc in mood_changes or []:
        entry = dict(mc)
        entry["type"] = "mood_change"
        timeline.append(entry)

    # Sort by time (speech segments by start, mood changes by timestamp)
    def _time_key(item: Dict[str, Any]) -> float:
        if item.get("type") == "mood_change":
            return float(item.get("timestamp", 0.0) or 0.0)
        return float(item.get("start", 0.0) or 0.0)

    timeline.sort(key=_time_key)
    return timeline
