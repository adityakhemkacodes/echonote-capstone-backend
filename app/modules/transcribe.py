# app/modules/transcribe.py

import os
import subprocess
import traceback
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import whisper
from pyannote.audio import Pipeline


def _ffmpeg_installed() -> bool:
    """Check if ffmpeg is installed and callable."""
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
    Use ffmpeg to extract mono 16kHz WAV audio from the video.
    Returns the path to the WAV file or None on failure.
    """
    if not _ffmpeg_installed():
        print("⚠ FFmpeg not available — cannot extract audio for diarization.")
        return None

    input_path = Path(video_path)
    audio_path = input_path.with_suffix(".diar.wav")

    # If we've already extracted it once, reuse it
    if audio_path.exists():
        return str(audio_path)

    try:
        print(f"Extracting audio for diarization to: {audio_path}")
        cmd = [
            "ffmpeg",
            "-y",           # overwrite
            "-i", str(video_path),
            "-ac", "1",     # mono
            "-ar", "16000", # 16 kHz
            "-vn",          # no video
            str(audio_path),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        if audio_path.exists():
            print("✓ Audio extraction complete.")
            return str(audio_path)
        else:
            print("✗ Audio extraction failed: output file not created.")
            return None

    except Exception as e:
        print(f"✗ Audio extraction error: {e}")
        traceback.print_exc()
        return None


def transcribe_with_speakers(video_path: str) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Convert meeting audio to:
      - full transcript text
      - speaker diarization segments
      - whisper segments with timestamps

    Returns:
        transcript_text: str
        speaker_segments: list[dict] like
          [{"start": float, "end": float, "speaker": "SPEAKER_0"}, ...]
        whisper_segments: list[dict] like
          [{"start": float, "end": float, "text": "..."}, ...]
    """
    print(f"Transcribing: {video_path}")

    # -------------------------------------------------------------
    # Step 0: Verify FFmpeg (required for Whisper + audio extraction)
    # -------------------------------------------------------------
    if not _ffmpeg_installed():
        print("✗ ERROR: FFmpeg is not installed or not in PATH.")
        print("  Whisper and diarization both require FFmpeg.")
        return "", [], []

    # -------------------------------------------------------------
    # Step 1: Whisper Transcription (CPU-only for stability)
    # -------------------------------------------------------------
    transcript_text = ""
    speaker_segments: List[Dict] = []
    whisper_segments: List[Dict] = []

    try:
        print("Loading Whisper model...")
        # Force CPU for stability (your GPU stack has been flaky)
        model = whisper.load_model("base", device="cpu")
        print("✓ Whisper loaded on CPU")

        result = model.transcribe(video_path, fp16=False)
        transcript_text = (result.get("text") or "").strip()

        # Collect time-stamped segments from Whisper
        raw_segments = result.get("segments") or []
        for seg in raw_segments:
            try:
                whisper_segments.append(
                    {
                        "start": float(seg.get("start", 0.0) or 0.0),
                        "end": float(seg.get("end", 0.0) or 0.0),
                        "text": (seg.get("text") or "").strip(),
                    }
                )
            except Exception:
                # If any weird segment, just skip it
                continue

        if not transcript_text:
            print("⚠ Whisper produced an EMPTY transcript — likely no speech or audio error.")
            return "", [], whisper_segments

        print(f"✓ Whisper transcription complete (words={len(transcript_text.split())})")

    except Exception:
        print("✗ Whisper transcription FAILED.")
        traceback.print_exc()
        return "", [], []

    # -------------------------------------------------------------
    # Step 2: Speaker Diarization (pyannote) — optional but preferred
    # -------------------------------------------------------------
    auth_token = os.getenv("PYANNOTE_AUTH_TOKEN")
    if not auth_token:
        print("⚠ No PYANNOTE_AUTH_TOKEN found — skipping diarization")
        return transcript_text, [], whisper_segments

    # Extract audio to WAV so pyannote doesn't choke on MP4
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

        # Pass the WAV path instead of MP4 to avoid Libsndfile MP4 issue
        diarization = pipeline(audio_path)

        segments: List[Dict] = []
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
    """
    Transcribe video to text only (ignore speaker info).
    """
    text, _, _ = transcribe_with_speakers(video_path)
    return text


def identify_speakers(video_path: str, transcript: str = None):
    """
    Identify speaker segments only (no full transcript returned).
    """
    _, speakers, _ = transcribe_with_speakers(video_path)
    return speakers

# Add to END of app/modules/transcribe.py

def transcribe_individual_audio(audio_path: str, participant_name: str = None):
    """
    Transcribe individual participant audio with optional known name.
    Returns: (transcript_text, segments_with_names)
    """
    print(f"Transcribing individual audio: {audio_path}")
    
    try:
        model = whisper.load_model("base", device="cpu")
        result = model.transcribe(audio_path, fp16=False)
        
        transcript_text = (result.get("text") or "").strip()
        
        # Add participant name to each segment
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", "").strip(),
                "speaker_name": participant_name,  # Direct name assignment
                "audio_file": audio_path
            })
        
        return transcript_text, segments
        
    except Exception as e:
        print(f"✗ Error transcribing {audio_path}: {e}")
        return "", []


def transcribe_multiple_audio_files(audio_files_with_names: List[Dict[str, str]]):
    """
    Transcribe multiple individual audio files.
    audio_files_with_names: [{"file": "path", "name": "John Doe"}, ...]
    Returns: (full_transcript, all_segments_with_speakers)
    """
    print(f"Transcribing {len(audio_files_with_names)} individual audio files...")
    
    all_segments = []
    full_texts = []
    
    for item in audio_files_with_names:
        text, segments = transcribe_individual_audio(
            item["file"], 
            participant_name=item.get("name")
        )
        full_texts.append(text)
        all_segments.extend(segments)
    
    # Sort segments by timestamp
    all_segments.sort(key=lambda x: x["start"])
    
    full_transcript = "\n".join(full_texts)
    
    return full_transcript, all_segments
