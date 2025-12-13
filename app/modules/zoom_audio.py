# app/modules/zoom_audio.py
from pathlib import Path
import re
from typing import List, Dict

_MEDIA_EXTS = {".mp4", ".m4a", ".aac", ".wav", ".mp3", ".ogg", ".flac"}


def _safe_speaker_id(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown"


def parse_speaker_from_zoom_audio_filename(file_name: str):
    """
    Example:
      audioadityakhemka153256432.mp4 -> ("aditya_khemka", "Adityakhemka")
    """
    fn = (file_name or "").strip()
    if not fn:
        return None

    base = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", fn)
    low = base.lower()
    if not low.startswith("audio"):
        return None

    s = base[5:].strip()
    if not s:
        return None

    s = re.sub(r"\d+$", "", s).strip()
    if not s:
        return None

    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None

    tokens = [t for t in s.split(" ") if t]
    speaker_name = " ".join(t[:1].upper() + t[1:] for t in tokens)
    speaker_id = _safe_speaker_id("_".join(t.lower() for t in tokens))
    return speaker_id, speaker_name


def build_speaker_registry_from_folder(folder_path: str) -> List[Dict]:
    """
    Returns:
      [
        {"speaker_id": "...", "name": "...", "audio_file": "..."},
        ...
      ]
    Skips 'audio<digits>.*' style files (likely full meeting audio).
    """
    p = Path(folder_path)
    if not p.exists() or not p.is_dir():
        return []

    out: List[Dict] = []
    seen = set()

    for f in p.iterdir():
        if not f.is_file() or f.suffix.lower() not in _MEDIA_EXTS:
            continue

        parsed = parse_speaker_from_zoom_audio_filename(f.name)
        if not parsed:
            continue

        # skip audio<digits> (no name)
        rest = f.stem[5:].strip()
        if re.fullmatch(r"\d+", rest):
            continue

        speaker_id, speaker_name = parsed
        if speaker_id in seen:
            continue
        seen.add(speaker_id)

        out.append({"speaker_id": speaker_id, "name": speaker_name, "audio_file": str(f)})

    return out
