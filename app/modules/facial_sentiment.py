# app/modules/facial_sentiment.py

from deepface import DeepFace
import cv2
from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple, List, Any


# -------------------------------------------------------------------
# Emotion mapping -> ONLY 3 classes: happy / neutral / sad
# -------------------------------------------------------------------
_DEEPFACE_TO_3 = {
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "angry": "sad",
    "fear": "sad",
    "disgust": "sad",
    "surprise": "neutral",  # noisy spikes; treat as neutral
}

def _to_3class(emotion: str) -> str:
    e = (emotion or "neutral").strip().lower()
    return _DEEPFACE_TO_3.get(e, "neutral")


def _normalize_pct(d: Dict[str, float]) -> Dict[str, float]:
    happy = float(d.get("happy", 0.0) or 0.0)
    neutral = float(d.get("neutral", 0.0) or 0.0)
    sad = float(d.get("sad", 0.0) or 0.0)
    s = happy + neutral + sad
    if s <= 0:
        return {"happy": 0.0, "neutral": 100.0, "sad": 0.0}
    return {
        "happy": round((happy / s) * 100.0, 2),
        "neutral": round((neutral / s) * 100.0, 2),
        "sad": round((sad / s) * 100.0, 2),
    }


def _calibrate_facial(raw_pct: Dict[str, float]) -> Dict[str, float]:
    """
    Tiny extra padding:
    - If neutral is clearly dominant, clamp sad a bit (DeepFace can overcall sad).
    """
    r = _normalize_pct(raw_pct)

    n = float(r.get("neutral", 0.0))
    s = float(r.get("sad", 0.0))
    h = float(r.get("happy", 0.0))

    # If neutral dominates, reduce sad slightly and give it mostly to neutral
    if n >= 65.0 and s > 0.0:
        cut = s * 0.25  # reduce sad by 25%
        s2 = s - cut
        n2 = n + cut * 0.80
        h2 = h + cut * 0.20
        return _normalize_pct({"happy": h2, "neutral": n2, "sad": s2})

    return r


def analyze_facial_sentiment(
    video_path: str,
    face_tracks: list = None,
    person_id_to_name: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      - emotions_timeline: [{timestamp, person_id, emotion, (optional) name}]
      - overall_bundle:
          {
            "happy": <calibrated %>,
            "neutral": <calibrated %>,
            "sad": <calibrated %>,
            "overall_raw": {...pct...},
            "overall_calibrated": {...pct...},
          }
    """
    print("Analyzing facial sentiment...")

    person_id_to_name = person_id_to_name or {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("✗ Could not open video for facial sentiment")
        raw = {"neutral": 100.0}
        cal = {"neutral": 100.0, "happy": 0.0, "sad": 0.0}
        return [], {"happy": 0.0, "neutral": 100.0, "sad": 0.0, "overall_raw": raw, "overall_calibrated": cal}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_sec = (frame_count / fps) if fps and frame_count else None

    # -------------------------------------------------------------------
    # If no face_tracks, fallback: uniform frame sampling across full video
    # -------------------------------------------------------------------
    if not face_tracks:
        print("No face tracks provided, sampling frames across entire video...")
        timeline, raw_pct = _analyze_from_frames(cap, fps=fps, duration_sec=duration_sec, num_samples=120)
        cap.release()
        cal_pct = _calibrate_facial(raw_pct)
        bundle = {
            "happy": cal_pct.get("happy", 0.0),
            "neutral": cal_pct.get("neutral", 100.0),
            "sad": cal_pct.get("sad", 0.0),
            "overall_raw": _normalize_pct(raw_pct),
            "overall_calibrated": cal_pct,
        }
        return timeline, bundle

    valid_tracks = [
        t for t in (face_tracks or [])
        if isinstance(t, dict) and ("timestamp" in t) and ("bbox" in t)
    ]
    if not valid_tracks:
        print("⚠ No valid face tracks, falling back to frame sampling")
        timeline, raw_pct = _analyze_from_frames(cap, fps=fps, duration_sec=duration_sec, num_samples=120)
        cap.release()
        cal_pct = _calibrate_facial(raw_pct)
        bundle = {
            "happy": cal_pct.get("happy", 0.0),
            "neutral": cal_pct.get("neutral", 100.0),
            "sad": cal_pct.get("sad", 0.0),
            "overall_raw": _normalize_pct(raw_pct),
            "overall_calibrated": cal_pct,
        }
        return timeline, bundle

    valid_tracks.sort(key=lambda x: float(x.get("timestamp", 0.0) or 0.0))

    # sample across meeting per person
    tracks_by_pid = defaultdict(list)
    for t in valid_tracks:
        pid = t.get("person_id") or "PERSON_0"
        tracks_by_pid[pid].append(t)

    MAX_TOTAL_ANALYSES = 240
    pids = list(tracks_by_pid.keys())
    per_person_budget = max(20, int(MAX_TOTAL_ANALYSES / max(1, len(pids))))

    def _uniform_sample(seq: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        if not seq:
            return []
        if len(seq) <= k:
            return seq
        step = (len(seq) - 1) / float(k - 1)
        idxs = [int(round(i * step)) for i in range(k)]
        out = []
        seen = set()
        for i in idxs:
            i = max(0, min(i, len(seq) - 1))
            if i in seen:
                continue
            seen.add(i)
            out.append(seq[i])
        return out

    selected_tracks: List[Dict[str, Any]] = []
    for pid, seq in tracks_by_pid.items():
        seq_sorted = sorted(seq, key=lambda x: float(x.get("timestamp", 0.0) or 0.0))
        selected_tracks.extend(_uniform_sample(seq_sorted, per_person_budget))

    selected_tracks.sort(key=lambda x: float(x.get("timestamp", 0.0) or 0.0))

    emotions_timeline: List[Dict[str, Any]] = []
    all_emotions_3: List[str] = []

    for track in selected_tracks:
        timestamp = float(track.get("timestamp", 0.0) or 0.0)

        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        try:
            x, y, w, h = track["bbox"]
            if w <= 0 or h <= 0:
                continue

            expand_factor = 1.35
            cx = x + w / 2.0
            cy = y + h / 2.0
            new_w = w * expand_factor
            new_h = h * expand_factor

            x1 = int(max(0, cx - new_w / 2.0))
            y1 = int(max(0, cy - new_h / 2.0))
            x2 = int(min(frame.shape[1], cx + new_w / 2.0))
            y2 = int(min(frame.shape[0], cy + new_h / 2.0))

            if x2 <= x1 or y2 <= y1:
                continue

            face_img = frame[y1:y2, x1:x2]
            if face_img is None or face_img.size == 0:
                continue

            result = DeepFace.analyze(
                face_img,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            if isinstance(result, list) and result:
                raw_emotion = result[0].get("dominant_emotion", "neutral")
            elif isinstance(result, dict):
                raw_emotion = result.get("dominant_emotion", "neutral")
            else:
                raw_emotion = "neutral"

            scores = None
            try:
                if isinstance(result, list) and result:
                    scores = result[0].get("emotion")
                elif isinstance(result, dict):
                    scores = result.get("emotion")
            except Exception:
                scores = None

            # dampen low-confidence
            if isinstance(scores, dict) and raw_emotion in scores:
                conf = float(scores.get(raw_emotion, 0.0) or 0.0) / 100.0
                if conf < 0.45:
                    raw_emotion = "neutral"

            emotion_3 = _to_3class(raw_emotion)

            pid = track.get("person_id") or "PERSON_0"
            entry: Dict[str, Any] = {
                "timestamp": float(timestamp),
                "person_id": pid,
                "emotion": emotion_3,
            }
            if pid in person_id_to_name and person_id_to_name[pid]:
                entry["name"] = person_id_to_name[pid]

            emotions_timeline.append(entry)
            all_emotions_3.append(emotion_3)

        except Exception:
            continue

    cap.release()

    raw_pct = _aggregate_emotions_3(all_emotions_3)
    cal_pct = _calibrate_facial(raw_pct)

    print(f"✓ Facial sentiment analysis complete: {len(emotions_timeline)} frames analyzed")

    bundle = {
        # keep top-level keys for backward compatibility (use calibrated)
        "happy": cal_pct.get("happy", 0.0),
        "neutral": cal_pct.get("neutral", 100.0),
        "sad": cal_pct.get("sad", 0.0),

        # also expose raw/cal explicitly for frontend
        "overall_raw": _normalize_pct(raw_pct),
        "overall_calibrated": cal_pct,
    }

    return emotions_timeline, bundle


def _analyze_from_frames(
    cap: cv2.VideoCapture,
    fps: float,
    duration_sec: Optional[float],
    num_samples: int = 120,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        return [], {"neutral": 100.0, "happy": 0.0, "sad": 0.0}

    num_samples = max(20, int(num_samples))
    frame_indices = sorted(
        set(int(i * (total_frames - 1) / (num_samples - 1)) for i in range(num_samples))
    )

    emotions_timeline: List[Dict[str, Any]] = []
    all_emotions_3: List[str] = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            if isinstance(result, list) and result:
                raw_emotion = result[0].get("dominant_emotion", "neutral")
            elif isinstance(result, dict):
                raw_emotion = result.get("dominant_emotion", "neutral")
            else:
                raw_emotion = "neutral"

            emotion_3 = _to_3class(raw_emotion)
            timestamp = float(frame_idx / (fps or 25.0))

            emotions_timeline.append(
                {"timestamp": timestamp, "person_id": "PERSON_0", "emotion": emotion_3}
            )
            all_emotions_3.append(emotion_3)

        except Exception:
            continue

    raw_pct = _aggregate_emotions_3(all_emotions_3)
    return emotions_timeline, raw_pct


def _aggregate_emotions_3(all_emotions_3: List[str]) -> Dict[str, float]:
    if not all_emotions_3:
        return {"happy": 0.0, "neutral": 100.0, "sad": 0.0}

    counts = Counter(all_emotions_3)
    total = sum(counts.values()) or 1

    return {
        "happy": round((counts.get("happy", 0) / total) * 100.0, 2),
        "neutral": round((counts.get("neutral", 0) / total) * 100.0, 2),
        "sad": round((counts.get("sad", 0) / total) * 100.0, 2),
    }
