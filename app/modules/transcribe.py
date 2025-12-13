# app/modules/transcribe.py

import os
import re
import subprocess
import traceback
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import whisper

try:
    from pyannote.audio import Pipeline  # type: ignore
except Exception:
    Pipeline = None


# --------------------------
# FFmpeg helpers
# --------------------------

def _ffmpeg_installed() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return True
    except Exception:
        return False


def _extract_audio_to_wav(video_path: str) -> Optional[str]:
    """
    Extract mono 16kHz WAV for pyannote (MP4 can fail via libsndfile).
    """
    if not _ffmpeg_installed():
        print("⚠ FFmpeg not available — cannot extract audio for diarization.")
        return None

    input_path = Path(video_path)
    audio_path = input_path.with_suffix(".diar.wav")

    if audio_path.exists():
        return str(audio_path)

    try:
        print(f"Extracting audio for diarization to: {audio_path}")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(audio_path),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        if audio_path.exists():
            print("✓ Audio extraction complete.")
            return str(audio_path)

        print("✗ Audio extraction failed: output file not created.")
        return None

    except Exception as e:
        print(f"✗ Audio extraction error: {e}")
        traceback.print_exc()
        return None


# --------------------------
# Zoom per-speaker detection
# --------------------------

# Accept common Zoom-ish audio sidecars
_AUDIO_EXTS = {".m4a", ".wav", ".aac", ".mp3"}

# Examples:
#   audio5731972797
#   audioAdityaKhemka75731972797
#   audioAdiRaje95731972797
#   audioAadiN85731972797
#
# We treat:
#   - audio<digits>          => full mix audio (NOT a speaker file)
#   - audio<Name><digits>    => per-speaker file
_AUDIO_STEM_RE = re.compile(r"^audio(?P<body>.*)$", re.IGNORECASE)
_TRAILING_DIGITS_RE = re.compile(r"(?P<name>.*?)(?P<digits>\d+)$")


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "unknown"


def _prettify_name(raw: str) -> str:
    """
    Turn things like:
      - "AdityaKhemka" -> "Aditya Khemka"
      - "AadiN"        -> "Aadi N"
      - "adi_raje"     -> "Adi Raje"
    """
    raw = (raw or "").strip()
    if not raw:
        return "Unknown"

    # normalize separators
    s = raw.replace("_", " ").replace("-", " ").strip()

    # split camelcase-ish: "AdiRaje" -> "Adi Raje"
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)

    # split letter-digit boundaries (rare but safe)
    s = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)
    s = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s)

    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()

    # Title-case words, but keep all-caps short tokens as-is
    parts = []
    for w in s.split():
        if len(w) <= 3 and w.isupper():
            parts.append(w)
        else:
            parts.append(w[:1].upper() + w[1:])
    return " ".join(parts).strip() or "Unknown"


def _parse_audio_stem(stem: str) -> Optional[Dict[str, str]]:
    """
    Parse filename stem (no extension), returning:
      {
        "kind": "speaker" | "full_mix",
        "speaker_raw": "...",   # only for kind="speaker"
        "meeting_id": "...."    # trailing digits
      }
    """
    m = _AUDIO_STEM_RE.match(stem or "")
    if not m:
        return None

    body = (m.group("body") or "").strip()
    if not body:
        return None

    m2 = _TRAILING_DIGITS_RE.match(body)
    if not m2:
        # No trailing digits => ambiguous, ignore
        return None

    name_part = (m2.group("name") or "").strip()
    digits = (m2.group("digits") or "").strip()
    if not digits:
        return None

    meeting_id = digits[-10:] if len(digits) > 10 else digits

    # If name_part is empty => this is audio<meeting_id> full mix
    if not name_part:
        return {"kind": "full_mix", "meeting_id": meeting_id}

    return {"kind": "speaker", "speaker_raw": name_part, "meeting_id": meeting_id}



def _find_zoom_audio_files(folder: Path) -> Dict[str, Any]:
    """
    Find Zoom sidecar audio in the same folder as the mp4.

    Returns:
      {
        "meeting_id": str,
        "full_mix": Optional[Path],
        "speakers": List[{"path": str, "speaker_name": str, "speaker_id": str, "meeting_id": str}]
      }

    We pick the meeting_id that appears most frequently among audio*.{m4a,wav,aac,mp3}.
    """
    if not folder.exists() or not folder.is_dir():
        return {"meeting_id": None, "full_mix": None, "speakers": []}

    parsed: List[Dict[str, Any]] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in _AUDIO_EXTS:
            continue
        stem = p.stem or ""
        if not stem.lower().startswith("audio"):
            continue

        info = _parse_audio_stem(stem)
        if not info:
            continue

        info["path"] = p
        parsed.append(info)

    if not parsed:
        return {"meeting_id": None, "full_mix": None, "speakers": []}

    # pick most common meeting_id
    counts: Dict[str, int] = {}
    for it in parsed:
        mid = it.get("meeting_id")
        if mid:
            counts[mid] = counts.get(mid, 0) + 1
    if not counts:
        return {"meeting_id": None, "full_mix": None, "speakers": []}

    meeting_id = max(counts.items(), key=lambda kv: kv[1])[0]

    # filter to that meeting_id
    parsed = [it for it in parsed if it.get("meeting_id") == meeting_id]

    full_mix: Optional[Path] = None
    speakers: List[Dict[str, str]] = []

    for it in parsed:
        kind = it.get("kind")
        p: Path = it.get("path")
        if kind == "full_mix":
            # choose the first full-mix if multiple
            if full_mix is None:
                full_mix = p
            continue

        if kind == "speaker":
            raw = str(it.get("speaker_raw") or "").strip()
            if not raw:
                continue
            speaker_name = _prettify_name(raw)
            speaker_id = f"SPEAKER_{_slugify(speaker_name)}"
            speakers.append(
                {
                    "path": str(p),
                    "speaker_name": speaker_name,
                    "speaker_id": speaker_id,
                    "meeting_id": meeting_id,
                }
            )

    # de-dupe speakers by speaker_id (keep first)
    seen = set()
    uniq: List[Dict[str, str]] = []
    for s in speakers:
        sid = s["speaker_id"]
        if sid in seen:
            continue
        seen.add(sid)
        uniq.append(s)

    # stable ordering: by speaker_name then filename
    uniq.sort(key=lambda x: (x.get("speaker_name") or "", Path(x.get("path") or "").name))

    return {"meeting_id": meeting_id, "full_mix": str(full_mix) if full_mix else None, "speakers": uniq}


# --------------------------
# Whisper helpers
# --------------------------

_WHISPER_MODEL = None


def _load_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        print("Loading Whisper model (base, CPU)...")
        _WHISPER_MODEL = whisper.load_model("base", device="cpu")
        print("✓ Whisper loaded on CPU")
    return _WHISPER_MODEL


def _transcribe_single_file(model, media_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Transcribe a single media file with Whisper.
    Returns (text, segments) where segments are [{start,end,text}]
    """
    result = model.transcribe(media_path, fp16=False)
    text = (result.get("text") or "").strip()

    segments_out: List[Dict[str, Any]] = []
    for seg in (result.get("segments") or []):
        try:
            segments_out.append(
                {
                    "start": float(seg.get("start", 0.0) or 0.0),
                    "end": float(seg.get("end", 0.0) or 0.0),
                    "text": (seg.get("text") or "").strip(),
                }
            )
        except Exception:
            continue

    return text, segments_out


# --------------------------
# Public API
# --------------------------

def transcribe_with_speakers(video_path: str) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Returns:
      transcript_text: str

      speaker_segments:
        - Zoom mode: [{start,end,text,speaker,speaker_id,speaker_name,source_file}, ...]
          (these are *utterance segments* already labeled with a speaker_id)
        - Legacy mode: diarization segments [{start,end,speaker}, ...]

      whisper_segments:
        - Zoom mode: [{start,end,text}, ...] (same as utterance segments, but without speaker fields)
        - Legacy mode: whisper segments [{start,end,text}, ...]
    """
    print(f"Transcribing: {video_path}")

    if not _ffmpeg_installed():
        print("✗ ERROR: FFmpeg is not installed or not in PATH.")
        return "", [], []

    video_p = Path(video_path)
    folder = video_p.parent

    # 1) Prefer Zoom per-speaker audio sidecars if present
    zoom = _find_zoom_audio_files(folder)
    zoom_speakers = zoom.get("speakers") or []

    # We only treat it as Zoom speaker mode if we have *speaker* files
    if zoom_speakers:
        print(f"✓ Found {len(zoom_speakers)} Zoom per-speaker audio files (meeting_id={zoom.get('meeting_id')}).")
        try:
            model = _load_whisper_model()

            full_transcript_parts: List[str] = []
            speaker_labeled_segments: List[Dict[str, Any]] = []
            combined_whisper_segments: List[Dict[str, Any]] = []

            for sp in zoom_speakers:
                sp_path = sp["path"]
                sp_name = sp["speaker_name"]
                sp_id = sp["speaker_id"]

                print(f"  - Transcribing: {Path(sp_path).name}  (speaker={sp_name})")
                text, segs = _transcribe_single_file(model, sp_path)

                if text:
                    # keep a readable transcript; processors use full_text anyway
                    full_transcript_parts.append(f"{sp_name}: {text}")

                for seg in segs:
                    t = (seg.get("text") or "").strip()
                    if not t:
                        continue

                    entry = {
                        "start": float(seg.get("start", 0.0) or 0.0),
                        "end": float(seg.get("end", 0.0) or 0.0),
                        "text": t,
                        "speaker": sp_id,  # legacy-friendly key
                        "speaker_id": sp_id,
                        "speaker_name": sp_name,
                        "source_file": str(Path(sp_path).name),
                    }
                    speaker_labeled_segments.append(entry)
                    combined_whisper_segments.append(
                        {"start": entry["start"], "end": entry["end"], "text": entry["text"]}
                    )

            transcript_text = "\n".join(full_transcript_parts).strip()

            # IMPORTANT:
            # For Zoom per-speaker exports, these timestamps are usually aligned to meeting time
            # (silence when the person isn't speaking). If a Zoom export doesn't align, you'll still
            # get correct per-speaker sentiment/action-items, but cross-speaker ordering may be noisy.
            return transcript_text, speaker_labeled_segments, combined_whisper_segments

        except Exception:
            print("⚠ Zoom per-speaker mode failed — falling back to legacy single-file transcription.")
            traceback.print_exc()

    # 2) Legacy: transcribe the main meeting video + optional pyannote diarization
    transcript_text = ""
    speaker_segments: List[Dict[str, Any]] = []
    whisper_segments: List[Dict[str, Any]] = []

    try:
        model = _load_whisper_model()
        result = model.transcribe(video_path, fp16=False)
        transcript_text = (result.get("text") or "").strip()

        for seg in (result.get("segments") or []):
            try:
                whisper_segments.append(
                    {
                        "start": float(seg.get("start", 0.0) or 0.0),
                        "end": float(seg.get("end", 0.0) or 0.0),
                        "text": (seg.get("text") or "").strip(),
                    }
                )
            except Exception:
                continue

        if not transcript_text:
            print("⚠ Whisper produced an EMPTY transcript — likely no speech or audio error.")
            return "", [], whisper_segments

        print(f"✓ Whisper transcription complete (words={len(transcript_text.split())})")

    except Exception:
        print("✗ Whisper transcription FAILED.")
        traceback.print_exc()
        return "", [], []

    auth_token = os.getenv("PYANNOTE_AUTH_TOKEN")
    if not auth_token or Pipeline is None:
        print("⚠ No PYANNOTE_AUTH_TOKEN (or pyannote missing) — skipping diarization")
        return transcript_text, [], whisper_segments

    audio_path = _extract_audio_to_wav(video_path)
    if not audio_path:
        print("⚠ Could not prepare audio for diarization — skipping speaker info.")
        return transcript_text, [], whisper_segments

    try:
        print("Loading pyannote audio diarization pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=auth_token,
        )

        diarization = pipeline(audio_path)

        segments: List[Dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                }
            )

        print(f"✓ Speaker diarization complete ({len(segments)} segments)")
        speaker_segments = segments

    except Exception:
        print("⚠ Diarization failed — continuing without speaker information.")
        traceback.print_exc()

    return transcript_text, speaker_segments, whisper_segments


def transcribe_video(video_path: str) -> str:
    text, _, _ = transcribe_with_speakers(video_path)
    return text


def identify_speakers(video_path: str, transcript: str = None):
    _, speakers, _ = transcribe_with_speakers(video_path)
    return speakers
