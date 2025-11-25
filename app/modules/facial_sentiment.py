# app/modules/facial_sentiment.py
from deepface import DeepFace
import cv2
from collections import Counter


def analyze_facial_sentiment(video_path: str, face_tracks: list = None):
    """
    Analyze facial emotions from video.
    face_tracks: list of dicts with {timestamp, bbox, person_id}

    Returns:
        - emotions_timeline: list of dicts {timestamp, person_id, emotion}
        - overall_sentiment: dict with % of each emotion
    """
    print("Analyzing facial sentiment...")

    # If no face tracks provided, sample frames instead
    if not face_tracks:
        print("No face tracks provided, sampling frames...")
        return _analyze_from_frames(video_path, num_samples=80)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("✗ Could not open video for facial sentiment")
        return [], {"neutral": 100.0}

    # More tracks + better sampling
    MAX_ANALYSES = 120

    sorted_tracks = sorted(
        [t for t in face_tracks if "timestamp" in t and "bbox" in t],
        key=lambda x: x["timestamp"],
    )

    if not sorted_tracks:
        print("⚠ No valid face tracks, falling back to frame sampling")
        cap.release()
        return _analyze_from_frames(video_path, num_samples=80)

    n_tracks = len(sorted_tracks)
    if n_tracks > MAX_ANALYSES:
        step = n_tracks / MAX_ANALYSES
        indices = [int(i * step) for i in range(MAX_ANALYSES)]
        selected_tracks = [sorted_tracks[i] for i in indices]
    else:
        selected_tracks = sorted_tracks

    emotions_timeline = []
    all_emotions = []

    for track in selected_tracks:
        timestamp = track["timestamp"]
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            x, y, w, h = track["bbox"]

            # Ensure valid bbox
            if w <= 0 or h <= 0:
                continue

            # expand bbox for better DeepFace performance
            expand_factor = 1.4
            cx = x + w / 2
            cy = y + h / 2
            new_w = w * expand_factor
            new_h = h * expand_factor

            x1 = int(max(0, cx - new_w / 2))
            y1 = int(max(0, cy - new_h / 2))
            x2 = int(min(frame.shape[1], cx + new_w / 2))
            y2 = int(min(frame.shape[0], cy + new_h / 2))

            if x2 <= x1 or y2 <= y1:
                continue

            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            result = DeepFace.analyze(
                face_img,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            if isinstance(result, list):
                emotion = result[0].get("dominant_emotion", "neutral")
            else:
                emotion = result.get("dominant_emotion", "neutral")

            emotions_timeline.append(
                {
                    "timestamp": float(timestamp),
                    "person_id": track.get("person_id", "Unknown"),
                    "emotion": emotion,
                }
            )
            all_emotions.append(emotion)

        except Exception:
            # Ignore per-frame errors
            continue

    cap.release()

    overall_sentiment = _aggregate_emotions(all_emotions)
    print(f"✓ Facial sentiment analysis complete: {len(emotions_timeline)} frames analyzed")
    return emotions_timeline, overall_sentiment


def _analyze_from_frames(video_path: str, num_samples: int = 80):
    """
    Analyze emotions by sampling frames uniformly from video.
    Fallback method when face tracking data is not available.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("✗ Could not open video for fallback facial sentiment")
        return [], {"neutral": 100.0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames == 0:
        cap.release()
        return [], {"neutral": 100.0}

    # Sample frames uniformly
    num_samples = max(10, num_samples)
    frame_indices = sorted(
        set(int(i * total_frames / num_samples) for i in range(num_samples))
    )

    emotions_timeline = []
    all_emotions = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            if isinstance(result, list):
                emotion = result[0].get("dominant_emotion", "neutral")
            else:
                emotion = result.get("dominant_emotion", "neutral")

            timestamp = frame_idx / fps

            emotions_timeline.append(
                {
                    "timestamp": float(timestamp),
                    "person_id": "PERSON_0",
                    "emotion": emotion,
                }
            )
            all_emotions.append(emotion)

        except Exception:
            continue

    cap.release()

    overall_sentiment = _aggregate_emotions(all_emotions)
    return emotions_timeline, overall_sentiment


def _aggregate_emotions(all_emotions):
    """
    Aggregate with smoothing / noise threshold:
    - always keep 'neutral'
    - always keep 'happy' (even if rare)
    - drop other emotions that appear very rarely
    """
    if not all_emotions:
        return {"neutral": 100.0}

    counts = Counter(all_emotions)
    total = sum(counts.values())

    MIN_FRAMES = 2  # allow emotions that appear at least twice
    filtered_counts = {}

    for emo, c in counts.items():
        # always keep neutral & happy
        if emo in ("neutral", "happy"):
            filtered_counts[emo] = c
            continue

        # for other emotions, drop if extremely rare
        if c < MIN_FRAMES:
            continue

        filtered_counts[emo] = c

    if not filtered_counts:
        filtered_counts = counts

    filtered_total = sum(filtered_counts.values())
    overall_sentiment = {
        emo: round((c / filtered_total) * 100.0, 2) for emo, c in filtered_counts.items()
    }
    return overall_sentiment
