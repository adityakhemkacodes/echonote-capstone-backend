# app/services/utils.py
import os
import re
from pathlib import Path
from typing import List


def clean_text(text: str):
    """Basic cleanup for raw transcripts."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_file_extension(filename: str):
    return os.path.splitext(filename)[1].lower()


# -----------------------------
# NEW helpers (Zoom-friendly)
# -----------------------------

_MEDIA_EXTS = {".mp4", ".m4a", ".aac", ".wav", ".mp3", ".ogg", ".flac"}


def list_media_files(folder: str) -> List[str]:
    """Return absolute paths of media files in a folder (non-recursive)."""
    out: List[str] = []
    try:
        p = Path(folder)
        if not p.exists() or not p.is_dir():
            return []
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in _MEDIA_EXTS:
                out.append(str(f.resolve()))
    except Exception:
        return []
    return out


def looks_like_zoom_speaker_audio(filename: str) -> bool:
    """
    True for files like:
      audioJohnDoe123.mp4
      audio_aditya_khemka_999.m4a
    False for "audio123456.mp4" (likely full meeting audio with no name).
    """
    fn = (filename or "").strip()
    if not fn:
        return False

    base = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", fn)
    if not base.lower().startswith("audio"):
        return False

    rest = base[5:].strip()
    if not rest:
        return False

    # remove trailing digits
    rest2 = re.sub(r"\d+$", "", rest).strip()
    if not rest2:
        return False

    # if what's left is purely digits -> not a named speaker file
    if re.fullmatch(r"\d+", rest.strip()):
        return False

    return True
