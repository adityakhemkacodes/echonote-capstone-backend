# app/modules/name_detection.py

import cv2
import pytesseract
from pytesseract import Output
import re
import spacy
import string
from collections import Counter, defaultdict

# Try to load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    print("⚠ spaCy model not loaded. Name detection will use simple rules.")


# -------------------------------------------------------------------
# Helpers / constants
# -------------------------------------------------------------------

# Common overlay or UI words that are NOT names but often appear capitalized
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

# Tokens we really don't want as part of names
_BAD_TOKENS = {
    "Sorry",
    "Anything",
    "Friday",
    "It’s",
    "It's",
    "I’ve",
    "I've",
    "Selenium",  # domain terms
    "MVP",
    "API",
    "UI",
    "KOs",
}


def _has_vowel(word: str) -> bool:
    return any(v in word.lower() for v in "aeiou")


def _looks_like_name_token(token: str) -> bool:
    """
    Basic heuristic: single token that could be part of a name.
    Used for OCR tokens.
    """
    t = token.strip()

    if not t:
        return False

    if len(t) < 3 or len(t) > 20:
        return False

    if not t.replace("-", "").isalpha():
        return False

    if not t[0].isupper():
        return False

    # Avoid all-caps (usually acronyms / UI labels)
    if t.isupper():
        return False

    if t in _NON_NAME_SINGLE_WORDS:
        return False

    if not _has_vowel(t):
        return False

    return True


def _clean_name_string(name: str) -> str:
    name = name.strip()
    name = name.strip(string.punctuation + " ")
    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)
    return name


# -------------------------------------------------------------------
# Plausible full-name check
# -------------------------------------------------------------------

def _is_plausible_name(candidate: str) -> bool:
    """
    Stricter check for a full name string (1–3 tokens).
    Used AFTER we've assembled a candidate like 'Adi Raje'.
    """
    candidate = _clean_name_string(candidate)
    if not candidate:
        return False

    parts = candidate.split()

    # 1–3 tokens: ['Adi'], ['Adi', 'Raje'], ['Adi', 'van', 'Something']
    if not (1 <= len(parts) <= 3):
        return False

    # total alphabetic length (no spaces) – avoid 3–4 letter junk like 'Ul Oe'
    total_len = len("".join(parts))
    if total_len < 5:
        return False

    # each word at least 3 chars – filters 'Ul', 'At', etc.
    if any(len(p) < 3 for p in parts):
        return False

    # reuse bad token filters
    if any(p in _BAD_TOKENS for p in parts):
        return False
    if any(p in _NON_NAME_SINGLE_WORDS for p in parts):
        return False

    return True


# ============================================================
# TESSERACT OCR RUNNER (single place)
# ============================================================

def _run_tesseract_ocr(roi_gray):
    """
    Input: grayscale ROI
    Output: list of raw strings (tokens / fragments) from OCR
    """
    texts = []

    data = pytesseract.image_to_data(
        roi_gray,
        output_type=Output.DICT,
        config="--oem 3 --psm 7",
    )

    t_texts = data.get("text", [])
    confs = data.get("conf", [])
    for i, raw in enumerate(t_texts):
        txt = (raw or "").strip()
        if not txt or len(txt) <= 1:
            continue

        # try to parse confidence
        try:
            c = int(float(confs[i]))
        except Exception:
            c = -1

        # slightly tolerant, we clean later
        if c < 60:
            continue

        texts.append(txt)

    return texts


# ====================================================================
# 1) Global OCR-based name extraction (bottom overlay band)
# ====================================================================

def extract_names_from_video(
    video_path: str,
    fps: int = 3,                 # sample a bit more often
    max_seconds: int = 180,
    bottom_fraction: float = 0.5, # slightly taller bottom band
):
    """
    Extract participant name tags using OCR from the bottom portion
    of the frame.

    Returns: list of dicts [{ "timestamp": float, "name": str }, ...]
    """
    print("Extracting names from video overlay (global band)...")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("✗ Could not open video for OCR name detection")
            return []

        frame_rate = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(frame_rate // fps))

        per_frame_names = []  # list of (timestamp, [names_on_frame])
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            timestamp = frame_idx / frame_rate
            if timestamp > max_seconds:
                break

            if frame_idx % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape

                # bottom X% of the frame
                start_row = int(h * (1.0 - bottom_fraction))
                start_row = max(0, min(start_row, h - 1))
                roi = gray[start_row:, :]

                # Upscale + enhance contrast for better small-text OCR
                scale_factor = 2.5
                roi_large = cv2.resize(
                    roi,
                    None,
                    fx=scale_factor,
                    fy=scale_factor,
                    interpolation=cv2.INTER_LINEAR,
                )
                roi_large = cv2.GaussianBlur(roi_large, (3, 3), 0)
                roi_large = cv2.equalizeHist(roi_large)

                frame_names = []

                try:
                    raw_tokens = _run_tesseract_ocr(roi_large)

                    # Try to assemble tokens into names
                    cleaned_tokens = [
                        t.strip(string.punctuation + " ")
                        for t in raw_tokens
                        if t.strip()
                    ]

                    i = 0
                    while i < len(cleaned_tokens):
                        t1 = cleaned_tokens[i]
                        if not _looks_like_name_token(t1):
                            i += 1
                            continue

                        candidate = None

                        if i + 1 < len(cleaned_tokens):
                            t2 = cleaned_tokens[i + 1]
                            # Try to form "First Last"
                            if _looks_like_name_token(t2):
                                candidate = f"{t1} {t2}"
                                i += 2
                            else:
                                candidate = t1
                                i += 1
                        else:
                            candidate = t1
                            i += 1

                        candidate = _clean_name_string(candidate)
                        if not _is_plausible_name(candidate):
                            continue

                        frame_names.append(candidate)

                except Exception as e:
                    print(f"⚠ OCR error on frame {frame_idx}: {e}")

                if frame_names:
                    per_frame_names.append((float(timestamp), frame_names))

            frame_idx += 1

        cap.release()

        all_name_tokens = []
        for ts, names in per_frame_names:
            all_name_tokens.extend(names)

        if not all_name_tokens:
            print("✓ Detected 0 overlay names from video")
            return []

        counter = Counter(all_name_tokens)

        # For now keep even single occurrences; higher threshold is possible later
        MIN_OCCURRENCE = 1
        stable_names = {n for n, c in counter.items() if c >= MIN_OCCURRENCE}

        results = []
        seen = set()
        for ts, names in per_frame_names:
            for n in names:
                if n in stable_names and n not in seen:
                    seen.add(n)
                    results.append({"timestamp": float(ts), "name": n})

        print(
            f"✓ Detected {len(stable_names)} stable overlay names from video: {list(stable_names)}"
        )
        return results

    except Exception as e:
        print(f"✗ Name extraction error: {e}")
        return []


# ====================================================================
# 2) Per-face OCR using face tracks (robust multi-person solution)
# ====================================================================

def extract_names_for_tracks(
    video_path: str,
    face_tracks: list,
    max_samples_per_person: int = 35,
):
    """
    Extract participant names by OCR'ing a label strip
    just BELOW each tracked face bounding box.

    More robust for multi-person meetings:
      - works for faces in any row/column of the grid
      - ties candidate names to real face tracks (PERSON_0, PERSON_1, ...)

    Returns a list of dicts:
      [{ "timestamp": float, "name": str }, ...]
    """
    print("Extracting names from video using face tracks...")

    # If we somehow have no tracks, fall back to the global method.
    if not face_tracks:
        print("⚠ No face tracks available, falling back to global OCR")
        return extract_names_from_video(video_path)

    try:
        tracks_by_pid = defaultdict(list)
        for t in face_tracks:
            pid = t.get("person_id")
            if pid is None:
                continue
            if "timestamp" not in t or "bbox" not in t:
                continue
            tracks_by_pid[pid].append(t)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("✗ Could not open video for per-face OCR")
            return []

        per_pid_name_votes = defaultdict(list)
        results = []

        for pid, tracks in tracks_by_pid.items():
            # sort by time
            tracks = sorted(tracks, key=lambda x: float(x.get("timestamp", 0.0) or 0.0))
            if not tracks:
                continue

            n = len(tracks)
            if n <= max_samples_per_person:
                sampled_tracks = tracks
            else:
                step = n / max_samples_per_person
                indices = [int(i * step) for i in range(max_samples_per_person)]
                sampled_tracks = [tracks[i] for i in indices]

            for tr in sampled_tracks:
                ts = float(tr.get("timestamp", 0.0) or 0.0)
                bbox = tr.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                h, w, _ = frame.shape
                x, y, bw, bh = bbox

                # Clamp bbox to frame
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                bw = max(1, min(bw, w - x))
                bh = max(1, min(bh, h - y))

                # Label strip below the face:
                #   start a bit below the chin, and go down up to ~2x face height
                label_top = y + int(bh * 0.6)
                label_bottom = y + int(bh * 2.0)

                label_top = max(0, min(label_top, h - 2))
                label_bottom = max(label_top + 1, min(label_bottom, h))

                x1 = max(0, int(x - bw * 0.3))
                x2 = min(w, int(x + bw * 1.3))

                if label_bottom <= label_top or x2 <= x1:
                    continue

                roi = frame[label_top:label_bottom, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Upscale & enhance
                scale_factor = 2.5
                roi_large = cv2.resize(
                    gray,
                    None,
                    fx=scale_factor,
                    fy=scale_factor,
                    interpolation=cv2.INTER_LINEAR,
                )
                roi_large = cv2.GaussianBlur(roi_large, (3, 3), 0)
                roi_large = cv2.equalizeHist(roi_large)

                frame_names = []
                try:
                    raw_tokens = _run_tesseract_ocr(roi_large)
                    cleaned_tokens = [
                        t.strip(string.punctuation + " ")
                        for t in raw_tokens
                        if t.strip()
                    ]

                    i = 0
                    while i < len(cleaned_tokens):
                        t1 = cleaned_tokens[i]
                        if not _looks_like_name_token(t1):
                            i += 1
                            continue

                        candidate = None

                        if i + 1 < len(cleaned_tokens):
                            t2 = cleaned_tokens[i + 1]
                            if _looks_like_name_token(t2):
                                candidate = f"{t1} {t2}"
                                i += 2
                            else:
                                candidate = t1
                                i += 1
                        else:
                            candidate = t1
                            i += 1

                        candidate = _clean_name_string(candidate)
                        if not _is_plausible_name(candidate):
                            continue

                        frame_names.append(candidate)

                except Exception as e:
                    print(f"⚠ OCR error at t={ts:.2f}s for {pid}: {e}")

                if frame_names:
                    for nm in frame_names:
                        per_pid_name_votes[pid].append(nm)
                        results.append({"timestamp": ts, "name": nm})

        cap.release()

        if not results:
            print("✓ Per-face OCR found 0 names")
        else:
            print("✓ Per-face OCR candidate names (filtered):")
            for pid, names in per_pid_name_votes.items():
                cnt = Counter(names)
                top = cnt.most_common(3)
                print(f"  - {pid}: {dict(top)}")

        return results

    except Exception as e:
        print(f"✗ Per-face name extraction error: {e}")
        # Fallback to global band if per-face fails
        return extract_names_from_video(video_path)


# ====================================================================
# 3) Name detection from transcript (NER + heuristics)
# ====================================================================

def detect_participant_names(transcript: str):
    """
    Extract person names from transcript using spaCy NER (if available),
    with a simple heuristic fallback.

    Returns a list of distinct name strings (e.g., ["John Doe", "Alice"]).
    """
    print("Detecting participant names from transcript...")
    transcript = (transcript or "").strip()

    if not transcript or len(transcript) < 10:
        print("⚠ Transcript too short for name detection")
        return []

    names = set()

    # -------------------------
    # Method 1: spaCy NER (preferred)
    # -------------------------
    if nlp:
        try:
            doc = nlp(transcript[:15000])
            for ent in doc.ents:
                if ent.label_ != "PERSON":
                    continue

                clean = _clean_name_string(ent.text)
                if not clean:
                    continue

                parts = clean.split()

                # Allow 1–3 tokens; skip overlong or obviously bad ones
                if len(parts) == 0 or len(parts) > 3:
                    continue

                # Filter junk tokens
                if any(p in _BAD_TOKENS for p in parts):
                    continue

                if any(p in _NON_NAME_SINGLE_WORDS for p in parts):
                    continue

                # Basic character length sanity check
                if not (3 <= len(clean) <= 40):
                    continue

                names.add(clean)

        except Exception as e:
            print(f"⚠ spaCy NER error: {e}")

    # -------------------------
    # Method 2: Simple fallback rules if spaCy finds nothing
    # -------------------------
    if not names:
        words = transcript.split()
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]

            w1_clean = w1.strip(string.punctuation)
            w2_clean = w2.strip(string.punctuation)

            if (
                w1_clean.istitle()
                and w2_clean.istitle()
                and w1_clean.isalpha()
                and w2_clean.isalpha()
                and len(w1_clean) > 1
                and len(w2_clean) > 1
                and _has_vowel(w1_clean)
                and _has_vowel(w2_clean)
            ):
                full_name = f"{w1_clean} {w2_clean}"
                parts = full_name.split()
                if any(p in _BAD_TOKENS for p in parts):
                    continue
                if any(p in _NON_NAME_SINGLE_WORDS for p in parts):
                    continue
                names.add(full_name)

    # Final cleaning + de-duplication
    cleaned = []
    for n in names:
        n = _clean_name_string(n)
        if not n:
            continue
        parts = n.split()
        if any(p in _BAD_TOKENS for p in parts):
            continue
        if any(p in _NON_NAME_SINGLE_WORDS for p in parts):
            continue
        cleaned.append(n)

    result = cleaned[:10]
    print(f"✓ Found {len(result)} potential participant names: {result}")
    return result
