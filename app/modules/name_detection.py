# app/modules/name_detection.py

import cv2
import numpy as np
import string
import re
from collections import defaultdict, Counter

# ==========================
#  Transcript name heuristics
# ==========================

try:
    import spacy

    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        _nlp = None
        print("⚠ spaCy model 'en_core_web_sm' not available, transcript name detection will use fallback rules.")
except Exception:
    _nlp = None
    print("⚠ spaCy not installed, transcript name detection will use fallback rules.")

# PaddleOCR backend (no RapidOCR, no ONNX here)
try:
    from paddleocr import PaddleOCR
except Exception as e:
    PaddleOCR = None
    print(f"✗ paddleocr not available: {e}")


_NON_NAME_SINGLE_WORDS = {
    "Please",
    "Recording",
    "Record",
    "Share",
    "Screen",
    "Zoom",
    "Meet",
    "Meeting",
    "Today",
    "Okay",
    "Ok",
    "Maybe",
    "Report",
    "Reports",
    "Host",
    "You",
    "Mute",
    "Unmute",
    "Live",
    "Chat",
    "View",
    "File",
    "Files",
    "Edit",
    "Help",
    "Done",
    "Next",
    "Back",
    "Retry",
    "Cancel",
    "Leave",
    "Join",
    "Laptop",
    "Guest",
}

_BAD_TOKENS = {
    "Sorry",
    "Anything",
    "Friday",
    "It’s",
    "It's",
    "I’ve",
    "I've",
    "Selenium",  # explicitly ban this as a "name"
    "MVP",
    "API",
    "UI",
    "KOs",
}


def _has_vowel(word: str) -> bool:
    return any(v in word.lower() for v in "aeiou")


def _clean_name_string(name: str) -> str:
    name = (name or "").strip()
    name = name.replace("\n", " ")
    name = name.strip(string.punctuation + " ")
    name = re.sub(r"\s+", " ", name)
    return name


def _looks_like_name_token(token: str) -> bool:
    t = token.strip()
    if not t:
        return False
    if len(t) < 2 or len(t) > 25:
        return False
    if not t.replace("-", "").isalpha():
        return False
    if not t[0].isupper():
        return False
    if t.isupper():
        return False
    if t in _NON_NAME_SINGLE_WORDS:
        return False
    if not _has_vowel(t):
        return False
    return True


def _is_plausible_full_name(candidate: str) -> bool:
    candidate = _clean_name_string(candidate)
    if not candidate:
        return False

    parts = candidate.split()
    if not (1 <= len(parts) <= 3):
        return False

    if len("".join(parts)) < 4:
        return False
    if any(len(p) < 2 for p in parts):
        return False
    if any(p in _BAD_TOKENS for p in parts):
        return False
    if any(p in _NON_NAME_SINGLE_WORDS for p in parts):
        return False
    if not any(_looks_like_name_token(p) for p in parts):
        return False

    return True


# ==========================
#  PaddleOCR wrapper
# ==========================

_paddle_ocr = None


def _get_paddle_ocr():
    """
    Lazy init PaddleOCR (English, CPU).
    If anything fails, we just disable OCR gracefully.
    """
    global _paddle_ocr
    if _paddle_ocr is None and PaddleOCR is not None:
        print("✓ Initializing PaddleOCR engine (English, CPU)...")
        try:
            _paddle_ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=False,
                show_log=False,
            )
        except Exception as e:
            print(f"✗ Failed to initialize PaddleOCR: {e}")
            _paddle_ocr = None
    return _paddle_ocr


def _run_paddle_ocr(bgr_image, scale_factor: float = 3.0):
    """
    Run PaddleOCR on a (possibly tiny) BGR ROI.
    Returns list of raw text strings (no name filtering).
    Fully wrapped in try/except so it never crashes the pipeline.
    """
    try:
        ocr = _get_paddle_ocr()
        if ocr is None:
            return []

        if bgr_image is None or bgr_image.size == 0:
            return []

        img = bgr_image
        if scale_factor and scale_factor != 1.0:
            h, w = img.shape[:2]
            if h == 0 or w == 0:
                return []
            img = cv2.resize(
                img,
                (int(w * scale_factor), int(h * scale_factor)),
                interpolation=cv2.INTER_CUBIC,
            )

        # PaddleOCR accepts BGR from cv2 directly
        result = None
        try:
            result = ocr.ocr(img, cls=False)
        except Exception as e:
            print(f"⚠ PaddleOCR call failed: {e}")
            return []

        texts = []
        # result is typically a list of [ [ [box], (text, score) ], ... ]
        for line in (result or []):
            if not isinstance(line, (list, tuple)):
                continue
            for det in line:
                if (
                    not isinstance(det, (list, tuple))
                    or len(det) < 2
                    or not isinstance(det[1], (list, tuple))
                    or len(det[1]) < 2
                ):
                    continue
                text = str(det[1][0]).strip()
                try:
                    score = float(det[1][1])
                except Exception:
                    score = 0.0
                if not text:
                    continue
                if score < 0.3:  # slightly lenient; we filter names later
                    continue
                texts.append(text)

        return texts

    except Exception as e:
        print(f"⚠ _run_paddle_ocr unexpected error: {e}")
        return []


# ==========================
#  ROI helpers
# ==========================

def _normalize_bbox(bbox, frame_w, frame_h):
    """
    Accepts either [x, y, w, h] or [x1, y1, x2, y2].
    Returns (x, y, w, h) clamped to frame.
    Completely defensive to avoid index errors.
    """
    try:
        if bbox is None:
            return None

        # If bbox is dict-like or np array, convert to list first
        if not isinstance(bbox, (list, tuple)):
            try:
                bbox = list(bbox)
            except Exception:
                return None

        if len(bbox) < 4:
            return None

        x0 = float(bbox[0])
        y0 = float(bbox[1])
        x1 = float(bbox[2])
        y1 = float(bbox[3])

        # Heuristic: if x1,y1 look like bottom-right coords
        if x1 > x0 and y1 > y0 and x1 <= frame_w + 5 and y1 <= frame_h + 5:
            x = max(0, int(round(x0)))
            y = max(0, int(round(y0)))
            w = max(1, int(round(x1 - x0)))
            h = max(1, int(round(y1 - y0)))
        else:
            # treat as (x, y, w, h)
            x = max(0, int(round(x0)))
            y = max(0, int(round(y0)))
            w = max(1, int(round(x1)))
            h = max(1, int(round(y1)))

        if x >= frame_w or y >= frame_h:
            return None

        if x + w > frame_w:
            w = max(1, frame_w - x)
        if y + h > frame_h:
            h = max(1, frame_h - y)

        return x, y, w, h
    except Exception as e:
        print(f"⚠ _normalize_bbox error: {e}, bbox={bbox}")
        return None


def _extract_nameplate_roi(frame, bbox):
    """
    Given a face bbox, crop a bottom-left band of that tile
    (where your labels are: bottom-left, small text).

    Tuned for your screenshot:
      - left ~65% of the tile
      - bottom ~30% of the tile
    """
    try:
        H, W, _ = frame.shape
    except Exception:
        return None

    norm = _normalize_bbox(bbox, W, H)
    if norm is None:
        return None

    x, y, w, h = norm

    # slightly generous region to cope with low-res/tiny text
    x1 = int(x + 0.02 * w)
    x2 = int(x + 0.65 * w)
    y1 = int(y + 0.70 * h)
    y2 = int(y + 1.00 * h)

    x1 = max(0, min(x1, W - 1))
    x2 = max(x1 + 1, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(y1 + 1, min(y2, H))

    roi = frame[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return None
    return roi


def _extract_names_from_roi(roi_bgr, debug_prefix=""):
    """
    Run OCR on an ROI and return plausible name strings,
    with strong guards around everything.
    """
    try:
        texts = _run_paddle_ocr(roi_bgr, scale_factor=3.5)

        if texts and debug_prefix:
            print(f"{debug_prefix} raw OCR texts: {texts}")

        names = []
        for raw in texts:
            cleaned = _clean_name_string(raw)
            if not cleaned:
                continue

            # strip trailing separators (e.g. "Adi Raje -", "Adi Raje | Zoom")
            cleaned = re.sub(r"[-|•·]+$", "", cleaned).strip()

            # split on separators and test each piece
            chunks = re.split(r"[|;/]", cleaned)
            for ch in chunks:
                ch = _clean_name_string(ch)
                if not ch:
                    continue
                if _is_plausible_full_name(ch):
                    names.append(ch)

        return names
    except Exception as e:
        print(f"⚠ _extract_names_from_roi error: {e}")
        return []


# ==========================
#  Per-face OCR (public)
# ==========================

def extract_names_for_tracks(video_path: str, face_tracks: list, max_samples_per_person: int = 30):
    """
    For each person_id in face_tracks, sample up to N timestamps,
    crop the bottom-left band of their tile, and run PaddleOCR on that ROI.
    Returns [{"person_id": ..., "name": ...}, ...]
    """
    print("📌 Running per-face OCR name extraction (PaddleOCR)...")

    if not face_tracks:
        print("⚠ No face tracks provided to per-face OCR.")
        return []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("✗ Could not open video for per-face OCR")
            return []

        tracks_by_pid = defaultdict(list)
        for tr in face_tracks:
            pid = tr.get("person_id")
            if pid is None:
                continue
            if "timestamp" not in tr or "bbox" not in tr:
                continue
            tracks_by_pid[pid].append(tr)

        name_votes = defaultdict(list)

        for pid, tracks in tracks_by_pid.items():
            tracks = sorted(tracks, key=lambda t: float(t.get("timestamp", 0.0) or 0.0))
            if not tracks:
                continue

            n = len(tracks)
            if n <= max_samples_per_person:
                sampled = tracks
            else:
                # safe index sampling
                step = n / max_samples_per_person
                idxs = [min(n - 1, int(i * step)) for i in range(max_samples_per_person)]
                sampled = [tracks[i] for i in idxs]

            for tr in sampled:
                try:
                    ts = float(tr.get("timestamp", 0.0) or 0.0)
                    bbox = tr.get("bbox")

                    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue

                    roi = _extract_nameplate_roi(frame, bbox)
                    if roi is None:
                        continue

                    detected_names = _extract_names_from_roi(
                        roi,
                        debug_prefix=f"[PID {pid} @ {ts:.2f}s]",
                    )

                    for nm in detected_names:
                        name_votes[pid].append(nm)
                except Exception as e:
                    print(f"⚠ Error OCR-ing track for pid={pid}: {e}")
                    continue

        cap.release()

        results = []
        print("📌 Per-face OCR name candidates:")
        for pid, values in name_votes.items():
            if not values:
                print(f"  - {pid}: (no names)")
                continue
            counter = Counter(values)
            top_name, top_count = counter.most_common(1)[0]
            print(f"  - {pid}: {dict(counter)} (selected: {top_name})")
            results.append({"person_id": pid, "name": top_name})

        if not results:
            print("✓ Per-face OCR did not find any stable names.")
        else:
            print(f"✓ Per-face OCR found {len(results)} unique names.")

        return results

    except Exception as e:
        print(f"⚠ extract_names_for_tracks unexpected error: {e}")
        try:
            cap.release()
        except Exception:
            pass
        print("✓ Per-face OCR did not find any names (due to error).")
        return []


# ==========================
#  Global bottom-band OCR
# ==========================

def extract_names_from_video(video_path: str, fps: int = 1, bottom_fraction: float = 0.30):
    """
    Fallback: scan the bottom band of each frame in 3 horizontal segments
    (left, middle, right) with PaddleOCR and aggregate name votes.
    Returns [{"name": ...}, ...] sorted by frequency.
    """
    print("📌 Running global bottom-band OCR (PaddleOCR)...")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("✗ Could not open video for global OCR")
            return []

        frame_rate = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(frame_rate // fps))

        names_seen = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx % frame_interval == 0:
                try:
                    H, W, _ = frame.shape
                except Exception:
                    frame_idx += 1
                    continue

                band_top = int(H * (1.0 - bottom_fraction))
                band_top = max(0, min(band_top, H - 2))
                band = frame[band_top:H, :]

                seg_w = max(1, W // 3)
                for k in range(3):
                    x1 = k * seg_w
                    x2 = W if k == 2 else (k + 1) * seg_w
                    roi = band[:, x1:x2]
                    if roi is None or roi.size == 0:
                        continue

                    det_names = _extract_names_from_roi(
                        roi,
                        debug_prefix=f"[Global seg {k} frame {frame_idx}]",
                    )
                    names_seen.extend(det_names)

            frame_idx += 1

        cap.release()

        if not names_seen:
            print("✓ Global OCR did not find any names.")
            return []

        counter = Counter(names_seen)
        print(f"✓ Global OCR raw counts: {dict(counter)}")

        ordered = [n for n, _ in counter.most_common()]
        return [{"name": n} for n in ordered]

    except Exception as e:
        print(f"⚠ extract_names_from_video unexpected error: {e}")
        try:
            cap.release()
        except Exception:
            pass
        print("✓ Global OCR did not find any names (due to error).")
        return []


# ==========================
#  Transcript-based names
# ==========================

def detect_participant_names(transcript: str):
    """
    Extract participant names from transcript using spaCy NER if available
    plus a simple capitalized-bigram fallback.
    Returns a list of unique strings.
    """
    print("📌 Detecting participant names from transcript...")
    transcript = (transcript or "").strip()

    if not transcript or len(transcript) < 10:
        print("⚠ Transcript too short for name detection")
        return []

    names = set()

    # spaCy NER path
    if _nlp is not None:
        try:
            doc = _nlp(transcript[:15000])
            for ent in doc.ents:
                if ent.label_ != "PERSON":
                    continue
                clean = _clean_name_string(ent.text)
                if not clean:
                    continue
                parts = clean.split()
                if len(parts) == 0 or len(parts) > 3:
                    continue
                if any(p in _BAD_TOKENS for p in parts):
                    continue
                if any(p in _NON_NAME_SINGLE_WORDS for p in parts):
                    continue
                if not (3 <= len(clean) <= 40):
                    continue
                names.add(clean)
        except Exception as e:
            print(f"⚠ spaCy NER error: {e}")

    # fallback: simple capitalised bigrams
    if not names:
        words = transcript.split()
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            w1c = w1.strip(string.punctuation)
            w2c = w2.strip(string.punctuation)
            if (
                w1c.istitle()
                and w2c.istitle()
                and w1c.isalpha()
                and w2c.isalpha()
                and _has_vowel(w1c)
                and _has_vowel(w2c)
                and len(w1c) > 1
                and len(w2c) > 1
            ):
                candidate = f"{w1c} {w2c}"
                if _is_plausible_full_name(candidate):
                    names.add(candidate)

    cleaned = []
    for n in names:
        n2 = _clean_name_string(n)
        if not n2:
            continue
        parts = n2.split()
        if any(p in _BAD_TOKENS for p in parts):
            continue
        if any(p in _NON_NAME_SINGLE_WORDS for p in parts):
            continue
        cleaned.append(n2)

    result = cleaned[:10]
    print(f"✓ Found {len(result)} potential participant names from transcript: {result}")
    return result