# app/modules/name_detection.py
import cv2
import pytesseract
from pytesseract import Output
import re
import spacy
import string
from collections import Counter

# Try to load spacy model for NER
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

    if len(t) < 2 or len(t) > 20:
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
# 1) OCR-based name extraction from video overlays
# -------------------------------------------------------------------

def extract_names_from_video(video_path: str, fps: int = 1):
    """
    Extract participant name tags using OCR from video overlays.

    We focus on the bottom band where platforms usually show name labels.
    Returns a list of dicts: [{timestamp, name}, ...]
    """
    print("Extracting names from video overlay...")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("✗ Could not open video for OCR name detection")
            return []

        frame_rate = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(frame_rate // fps))

        # Collect raw names per frame to later compute stability
        per_frame_names = []  # list of (timestamp, [names_on_frame])
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_idx % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape

                # Bottom ~20% of the frame – typical name strip
                bottom_crop = gray[int(h * 0.8) :, :]

                frame_names = []

                try:
                    data = pytesseract.image_to_data(
                        bottom_crop,
                        output_type=Output.DICT,
                        config="--oem 3 --psm 6",
                    )
                    text_blocks = [
                        t.strip() for t in data["text"] if t and len(t.strip()) > 1
                    ]

                    cleaned_tokens = [
                        t.strip(string.punctuation + " ") for t in text_blocks
                    ]

                    i = 0
                    while i < len(cleaned_tokens):
                        t1 = cleaned_tokens[i]
                        if not _looks_like_name_token(t1):
                            i += 1
                            continue

                        # Try to pair with next token into "First Last"
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
                        if not candidate:
                            continue

                        parts = candidate.split()
                        if any(p in _BAD_TOKENS for p in parts):
                            continue

                        # Limit to 1–3 words
                        if not (1 <= len(parts) <= 3):
                            continue

                        frame_names.append(candidate)

                except Exception as e:
                    print(f"⚠ OCR error on frame {frame_idx}: {e}")

                timestamp = frame_idx / frame_rate
                if frame_names:
                    per_frame_names.append((timestamp, frame_names))

            frame_idx += 1

            # Limit processing to first ~10 seconds for speed
            if frame_idx > int(frame_rate * 10):
                break

        cap.release()

        # ---- stability filtering: keep names that appear in multiple frames ----
        all_name_tokens = []
        for ts, names in per_frame_names:
            all_name_tokens.extend(names)

        if not all_name_tokens:
            print("✓ Detected 0 overlay names from video")
            return []

        counter = Counter(all_name_tokens)

        MIN_OCCURRENCE = 2
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


# -------------------------------------------------------------------
# 2) Name detection from transcript (NER + heuristics)
# -------------------------------------------------------------------

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
