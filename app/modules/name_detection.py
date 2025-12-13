# app/modules/name_detection.py
import cv2
import string
import re
from collections import defaultdict, Counter
from typing import List, Dict, Optional

try:
    import spacy  # type: ignore
    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        _nlp = None
        print("⚠ spaCy model 'en_core_web_sm' not available, transcript name detection will use fallback rules.")
except Exception:
    _nlp = None
    print("⚠ spaCy not installed, transcript name detection will use fallback rules.")

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None
    print("⚠ PaddleOCR not available (optional). OCR-based name detection will be disabled.")


_NON_NAME_SINGLE_WORDS = {
    "Please","Recording","Record","Share","Screen","Zoom","Meet","Meeting","Today","Okay","Ok","Maybe",
    "Report","Reports","Host","You","Mute","Unmute","Live","Chat","View","File","Files","Edit","Help",
    "Done","Next","Back","Retry","Cancel","Leave","Join","Laptop","Guest",
}

_BAD_TOKENS = {
    "Sorry","Anything","Friday","It’s","It's","I’ve","I've","Selenium","MVP","API","UI","KOs",
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
    """
    True for plausible name-ish tokens.
    NOTE: we allow 1-letter tokens ONLY if they are an uppercase initial (e.g. "N").
    """
    t = (token or "").strip()
    if not t:
        return False

    # allow single-letter initials (N, R, J, etc.)
    if len(t) == 1:
        return t.isalpha() and t.isupper()

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
    """
    More permissive than before:
      - allow last token to be a single-letter initial (e.g. "Aadi N")
      - still blocks junk tokens and obvious UI words
    """
    candidate = _clean_name_string(candidate)
    if not candidate:
        return False

    parts = candidate.split()
    if not (1 <= len(parts) <= 4):
        return False

    if len("".join(parts)) < 3:
        return False

    # allow single-letter only if it's the LAST token and uppercase
    for i, p in enumerate(parts):
        if len(p) < 2:
            if not (i == len(parts) - 1 and len(p) == 1 and p.isalpha() and p.isupper()):
                return False

    if any(p in _BAD_TOKENS for p in parts):
        return False
    if any(p in _NON_NAME_SINGLE_WORDS for p in parts):
        return False

    # at least one token must look name-ish
    if not any(_looks_like_name_token(p) for p in parts):
        return False

    return True


# ============================================================
# Zoom per-speaker audio filename parsing (primary)
# ============================================================

_AUDIO_SPEAKER_RE = re.compile(r"^audio(?P<name>.*?)(?P<digits>\d+)?$", re.IGNORECASE)


def _split_camel(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = s.replace("_", " ").replace("-", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _slugify_like_transcribe(display_name: str) -> str:
    s = (display_name or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "unknown"


def extract_names_from_zoom_audio_filenames(file_names: List[str]) -> List[Dict]:
    out: List[Dict] = []
    seen = set()

    for fn in file_names or []:
        try:
            base = (fn or "").strip()
            if not base:
                continue

            base_no_ext = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", base)
            if not base_no_ext.lower().startswith("audio"):
                continue

            m = _AUDIO_SPEAKER_RE.match(base_no_ext)
            if not m:
                continue

            raw_name = (m.group("name") or "").strip()
            if not raw_name:
                continue

            raw_name = re.sub(r"\d+$", "", raw_name).strip()
            if not raw_name:
                continue

            cooked = _split_camel(raw_name)
            cooked = _clean_name_string(cooked)
            if not cooked:
                continue

            parts = cooked.split()
            display_parts = []
            for p in parts:
                if len(p) == 1:
                    display_parts.append(p.upper())
                else:
                    display_parts.append(p[:1].upper() + p[1:])
            display = " ".join(display_parts).strip()
            if not display:
                continue

            speaker_id = f"SPEAKER_{_slugify_like_transcribe(display)}"
            key = (speaker_id, display)
            if key in seen:
                continue
            seen.add(key)

            out.append({"speaker_id": speaker_id, "name": display, "file_name": fn})
        except Exception:
            continue

    return out


# ==========================
# PaddleOCR wrapper (optional)
# ==========================

_paddle_ocr = None


def _get_paddle_ocr():
    global _paddle_ocr
    if PaddleOCR is None:
        return None
    if _paddle_ocr is None:
        try:
            print("✓ Initializing PaddleOCR engine (English, CPU)...")
            _paddle_ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=False,
                show_log=False,
            )
        except Exception as e:
            print(f"⚠ Failed to initialize PaddleOCR (disabling OCR): {e}")
            _paddle_ocr = None
    return _paddle_ocr


def _run_paddle_ocr(bgr_image, scale_factor: float = 3.0):
    ocr = _get_paddle_ocr()
    if ocr is None:
        return []

    try:
        if bgr_image is None or getattr(bgr_image, "size", 0) == 0:
            return []

        img = bgr_image
        if scale_factor and scale_factor != 1.0:
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                return []
            img = cv2.resize(
                img,
                (int(w * scale_factor), int(h * scale_factor)),
                interpolation=cv2.INTER_CUBIC,
            )

        result = ocr.ocr(img, cls=False)

        texts = []
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
                if not text or score < 0.3:
                    continue
                texts.append(text)

        return texts

    except Exception as e:
        print(f"⚠ PaddleOCR error (ignored): {e}")
        return []


# ==========================
# ROI helpers + OCR public APIs (UNCHANGED)
# ==========================

def _normalize_bbox(bbox, frame_w, frame_h):
    try:
        if bbox is None:
            return None
        if not isinstance(bbox, (list, tuple)):
            try:
                bbox = list(bbox)
            except Exception:
                return None
        if len(bbox) < 4:
            return None

        x0 = float(bbox[0]); y0 = float(bbox[1])
        x1 = float(bbox[2]); y1 = float(bbox[3])

        if x1 > x0 and y1 > y0 and x1 <= frame_w + 5 and y1 <= frame_h + 5:
            x = max(0, int(round(x0)))
            y = max(0, int(round(y0)))
            w = max(1, int(round(x1 - x0)))
            h = max(1, int(round(y1 - y0)))
        else:
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
    except Exception:
        return None


def _extract_nameplate_roi(frame, bbox):
    try:
        H, W, _ = frame.shape
    except Exception:
        return None

    norm = _normalize_bbox(bbox, W, H)
    if norm is None:
        return None

    x, y, w, h = norm

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
    texts = _run_paddle_ocr(roi_bgr, scale_factor=3.5)
    if not texts:
        return []

    if debug_prefix:
        print(f"{debug_prefix} raw OCR texts: {texts}")

    names = []
    for raw in texts:
        cleaned = _clean_name_string(raw)
        if not cleaned:
            continue

        cleaned = re.sub(r"[-|•·]+$", "", cleaned).strip()
        chunks = re.split(r"[|;/]", cleaned)
        for ch in chunks:
            ch = _clean_name_string(ch)
            if not ch:
                continue
            if _is_plausible_full_name(ch):
                names.append(ch)

    return names


def extract_names_for_tracks(video_path: str, face_tracks: list, max_samples_per_person: int = 30):
    if PaddleOCR is None:
        return []

    print("📌 Running per-face OCR name extraction (PaddleOCR)...")

    if not face_tracks:
        return []

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
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
                step = n / max_samples_per_person
                idxs = [min(n - 1, int(i * step)) for i in range(max_samples_per_person)]
                sampled = [tracks[i] for i in idxs]

            for tr in sampled:
                ts = float(tr.get("timestamp", 0.0) or 0.0)
                bbox = tr.get("bbox")

                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                roi = _extract_nameplate_roi(frame, bbox)
                if roi is None:
                    continue

                detected_names = _extract_names_from_roi(roi)
                for nm in detected_names:
                    name_votes[pid].append(nm)

        results = []
        for pid, values in name_votes.items():
            if not values:
                continue
            counter = Counter(values)
            top_name, _ = counter.most_common(1)[0]
            results.append({"person_id": pid, "name": top_name})

        return results

    except Exception:
        return []
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def extract_names_from_video(video_path: str, fps: int = 1, bottom_fraction: float = 0.30):
    if PaddleOCR is None:
        return []

    print("📌 Running global bottom-band OCR (PaddleOCR)...")

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
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
                H, W = frame.shape[:2]
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
                    names_seen.extend(_extract_names_from_roi(roi))

            frame_idx += 1

        if not names_seen:
            return []

        counter = Counter(names_seen)
        ordered = [n for n, _ in counter.most_common()]
        return [{"name": n} for n in ordered]

    except Exception:
        return []
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def detect_participant_names(transcript: str):
    print("📌 Detecting participant names from transcript...")
    transcript = (transcript or "").strip()

    if not transcript or len(transcript) < 10:
        return []

    names = set()

    if _nlp is not None:
        try:
            doc = _nlp(transcript[:15000])
            for ent in doc.ents:
                if ent.label_ != "PERSON":
                    continue
                clean = _clean_name_string(ent.text)
                if not clean:
                    continue
                if not (3 <= len(clean) <= 40):
                    continue
                if _is_plausible_full_name(clean):
                    names.add(clean)
        except Exception as e:
            print(f"⚠ spaCy NER error: {e}")

    # Fallback: TitleCase pairs OR TitleCase + Initial
    words = transcript.split()
    for i in range(len(words) - 1):
        w1 = words[i].strip(string.punctuation)
        w2 = words[i + 1].strip(string.punctuation)

        if not w1 or not w2:
            continue

        # Allow "Aadi N"
        if w1.istitle() and (w2.isupper() and len(w2) == 1):
            candidate = f"{w1} {w2}"
            if _is_plausible_full_name(candidate):
                names.add(candidate)

        # Allow "Adi Raje"
        if w1.istitle() and w2.istitle() and w1.isalpha() and w2.isalpha():
            candidate = f"{w1} {w2}"
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

    # stable + short list
    cleaned = sorted(set(cleaned), key=lambda x: (len(x.split()), len(x)), reverse=True)
    result = cleaned[:10]
    print(f"✓ Found {len(result)} potential participant names from transcript: {result}")
    return result
