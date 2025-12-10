# app/modules/summarizer.py
"""
Gemini-based summarizer + action item extractor.

- generate_summary(...): uses Gemini to produce an executive summary + key topics
- extract_action_items(...): uses Gemini to extract structured action items

Both functions keep the same public signatures as the previous implementation,
so MeetingProcessor and the rest of the pipeline can stay unchanged.
"""

import os
import re
import json
import warnings
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Gemini SDK import
# -------------------------------------------------------------------

_GEMINI_AVAILABLE = False
_GEMINI_IMPORT_ERROR = None

try:
    import google.generativeai as genai  # type: ignore

    _GEMINI_AVAILABLE = True
except Exception as e:  # pragma: no cover
    _GEMINI_AVAILABLE = False
    _GEMINI_IMPORT_ERROR = e


# -------------------------------------------------------------------
# API key loading / model helper
# -------------------------------------------------------------------

def _load_gemini_api_key() -> Optional[str]:
    """
    Load Gemini API key from:
      1) GEMINI_API_KEY
      2) GOOGLE_API_KEY
      3) .gemini_key file in app/ or project root

    This avoids Windows env-var weirdness: you can just create a `.gemini_key`
    file with your key in it if you prefer.
    """
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key.strip()

    # Fallback: look for a .gemini_key file
    try:
        here = Path(__file__).resolve()
        candidates = [
            here.parent / ".gemini_key",          # app/modules/.gemini_key
            here.parent.parent / ".gemini_key",   # app/.gemini_key
            here.parent.parent.parent / ".gemini_key",  # project root/.gemini_key
        ]
        for p in candidates:
            if p.is_file():
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    return content
    except Exception:
        pass

    return None


def _get_gemini_model(model_name: Optional[str] = None):
    """
    Lazy-configures Gemini and returns a GenerativeModel instance.

    Model resolution order:
      1. Explicit `model_name` argument
      2. GEMINI_MODEL env var
      3. Default: "models/gemini-pro-latest"

    You can see available models for your key by running list_gemini_models.py.
    """
    if not _GEMINI_AVAILABLE:
        raise RuntimeError(f"Gemini SDK not available: {_GEMINI_IMPORT_ERROR}")

    api_key = _load_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key missing. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
            "or create a .gemini_key file with your API key."
        )

    genai.configure(api_key=api_key)

    # Prefer explicit arg, then env, then default to a REAL model id
    raw_name = model_name or os.getenv("GEMINI_MODEL") or "models/gemini-pro-latest"
    raw_name = (raw_name or "").strip()

    # Simple aliases for older-style names if you ever use them
    alias_map = {
        "gemini-pro": "models/gemini-pro-latest",
        "gemini-flash": "models/gemini-flash-latest",
        "gemini-flash-latest": "models/gemini-flash-latest",
        "gemini-pro-latest": "models/gemini-pro-latest",
    }
    effective_name = alias_map.get(raw_name, raw_name)

    print(f"[Gemini] Using model: {effective_name}")
    return genai.GenerativeModel(effective_name)




# -------------------------------------------------------------------
# Basic text helpers
# -------------------------------------------------------------------

_NOISE_KEYWORDS = [
    "coffee",
    "tea",
    "lunch",
    "snack",
    "pizza",
    "mascot",
    "cat ",
    "cats ",
    "joke",
    "laugh",
    "crying",
    "dark mode powered by chaos",
    "coffee machine",
    "construction noise",
    "wifi",
    "wi-fi",
    "headset",
]


def _split_into_sentences(text: str) -> List[str]:
    """Crude sentence splitter, good enough for transcripts."""
    text = (text or "").strip()
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


def _is_noise_sentence(s: str) -> bool:
    lower = s.lower()
    return any(k in lower for k in _NOISE_KEYWORDS)


def _clean_transcript_for_summarization(text: str) -> str:
    """
    Remove obvious small talk / setup noise so the summarizer
    focuses on the actual meeting content.
    """
    sentences = _split_into_sentences(text)
    cleaned: List[str] = []

    for s in sentences:
        if len(s) < 15:
            # ultra-short fragments are usually ASR noise or fillers
            continue

        lower = s.lower()

        # greetings / closings / pure chit-chat
        if any(
            p in lower
            for p in [
                "thanks for joining",
                "can you hear me",
                "hope you can both hear me",
                "sorry",
                "apologies",
                "loud and clear",
                "anyone else",
                "anything else important",
                "meeting adjourned",
                "see you",
            ]
        ):
            continue

        if _is_noise_sentence(s):
            continue

        cleaned.append(s)

    # fallback: if we stripped too aggressively, use original text
    if len(" ".join(cleaned).split()) < 30:
        return text.strip()

    return " ".join(cleaned)


def _safe_json_from_text(raw: str) -> Optional[Any]:
    """
    Try to recover a JSON object/array from a model response that *should*
    be JSON but may have extra text around it.
    """
    raw = (raw or "").strip()

    # If it's already pure JSON, try directly
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try to extract the first {...} or [...] block
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    first_bracket = raw.find("[")
    last_bracket = raw.rfind("]")

    candidates = []
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(raw[first_brace : last_brace + 1])
    if first_bracket != -1 and last_bracket > first_bracket:
        candidates.append(raw[first_bracket : last_bracket + 1])

    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue

    return None


# -------------------------------------------------------------------
# Gemini prompt helpers
# -------------------------------------------------------------------

def _gemini_meeting_summary_call(
    transcript: str,
    sentiment_data: Optional[Dict[str, Any]],
    topics: Optional[Dict[str, Any]],
    participants: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ask Gemini for a structured meeting summary.

    Expected JSON shape:

    {
      "main_summary": "string",
      "key_topics": ["topic 1", "topic 2", ...]
    }
    """
    model = _get_gemini_model()

    meta_bits = []

    # sentiment
    if sentiment_data and isinstance(sentiment_data, dict):
        overall = (
            sentiment_data.get("overall_mood")
            or sentiment_data.get("overall")
            or {}
        )
        if overall:
            meta_bits.append(f"Heuristic overall sentiment: {overall}")

    # participants
    if participants and isinstance(participants, dict):
        count = participants.get("count") or participants.get("participant_count")
        names = participants.get("detected_names") or participants.get("names") or []
        if count:
            meta_bits.append(f"Estimated participant count: {count}")
        if names:
            meta_bits.append("Detected participant names: " + ", ".join(names))

    # topics
    if topics and isinstance(topics, dict) and topics.get("topics"):
        top_summaries = []
        for t in topics["topics"][:3]:
            ts = (t.get("summary") or t.get("title") or "").strip()
            if ts:
                top_summaries.append(ts)
        if top_summaries:
            meta_bits.append("Heuristic topic summaries: " + " | ".join(top_summaries))

    meta_context = "\n".join(meta_bits) if meta_bits else "(no extra context)"

    system_instructions = textwrap.dedent(
        """
        You are an expert meeting summariser for business/product/engineering meetings.
        Your job is to give a concise, information-dense executive summary, not a transcript.
        Focus on:
        - main goals and outcomes
        - key decisions and tradeoffs
        - major risks, blockers, and uncertainties
        - follow-up work and owners if clear

        Ignore: greetings, small talk, jokes, meta comments about microphones, screens,
        or people joining/leaving the call.
        """
    ).strip()

    user_prompt = textwrap.dedent(
        f"""
        {system_instructions}

        Additional context (may be noisy, use only if helpful):
        {meta_context}

        Transcript:
        \"\"\" 
        {transcript}
        \"\"\" 

        Return ONLY valid JSON with this exact structure and no other text, no markdown:

        {{
          "main_summary": "A clear, 1–3 paragraph executive summary of the meeting.",
          "key_topics": [
            "Short bullet-style topic summary 1",
            "Short bullet-style topic summary 2"
          ]
        }}
        """
    ).strip()

    response = model.generate_content(user_prompt)
    raw_text = getattr(response, "text", "") or ""
    parsed = _safe_json_from_text(raw_text)

    if not isinstance(parsed, dict):
        raise ValueError("Gemini did not return a valid JSON object for summary.")

    main_summary = str(parsed.get("main_summary") or "").strip()
    key_topics = parsed.get("key_topics") or []
    if not isinstance(key_topics, list):
        key_topics = [str(key_topics)]

    key_topics = [str(t).strip() for t in key_topics if str(t).strip()]

    if not main_summary:
        main_summary = (
            transcript[:500] + "..." if len(transcript) > 500 else transcript
        )

    return {
        "main_summary": main_summary,
        "key_topics": key_topics,
    }


def _gemini_action_items_call(
    transcript: str,
    speaker_segments: Optional[List[Dict[str, Any]]] = None,
    participants: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Ask Gemini to extract structured action items.

    We ask Gemini to return a JSON array of objects:

    [
      {
        "action": "concrete action phrased as a task",
        "role": "action | decision | blocker",
        "confidence": "low | medium | high",
        "assignee": "Name of person responsible or null if unknown"
      },
      ...
    ]
    """
    model = _get_gemini_model()

    # Give Gemini some participant names to help assign owners
    participant_names: List[str] = []
    if participants and isinstance(participants, dict):
        participant_names = participants.get("detected_names") or []
        if not participant_names:
            plist = (
                participants.get("participants")
                or participants.get("participants_fused")
                or []
            )
            for p in plist:
                nm = p.get("canonical_name") or p.get("name")
                if nm:
                    participant_names.append(str(nm))
    names_hint = ", ".join(participant_names) if participant_names else "unknown"

    user_prompt = textwrap.dedent(
        f"""
        You are an expert at extracting actionable follow-ups from meeting transcripts.

        From the transcript below, extract:
        - clear action items
        - important decisions
        - any explicit blockers/risks that require follow-up

        Focus ONLY on meaningful work: implementation tasks, fixes, follow-ups,
        emails to send, documents to prepare, investigations to run, deployments,
        feature changes, scheduling meetings, etc.

        Ignore small talk, commentary, vague opinions, and generic statements
        that do not result in a concrete action or decision.

        Where possible, assign each item to a person using the names
        mentioned in the transcript and this approximate participant list:
        {names_hint}

        Transcript:
        \"\"\" 
        {transcript}
        \"\"\" 

        Return ONLY valid JSON: an array of objects with this exact schema:

        [
          {{
            "action": "Concrete, standalone description of the task or decision.",
            "role": "action" | "decision" | "blocker",
            "confidence": "low" | "medium" | "high",
            "assignee": "Name of responsible person or null if not clear"
          }}
        ]

        Do not include any explanation or markdown, just the JSON array.
        """
    ).strip()

    response = model.generate_content(user_prompt)
    raw_text = getattr(response, "text", "") or ""
    parsed = _safe_json_from_text(raw_text)

    if not isinstance(parsed, list):
        raise ValueError("Gemini did not return a JSON array for action items.")

    results: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        action = str(item.get("action") or "").strip()
        if not action:
            continue

        role = str(item.get("role") or "action").strip().lower()
        if role not in ("action", "decision", "blocker"):
            role = "action"

        confidence = str(item.get("confidence") or "medium").strip().lower()
        if confidence not in ("low", "medium", "high"):
            confidence = "medium"

        assignee = item.get("assignee")
        if isinstance(assignee, str):
            assignee = assignee.strip() or None
        else:
            assignee = None

        results.append(
            {
                "action": action,
                "role": role,
                "confidence": confidence,
                "assignee": assignee,
            }
        )

    return results


# -------------------------------------------------------------------
# PUBLIC API: generate_summary
# -------------------------------------------------------------------

def generate_summary(
    transcript: str,
    sentiment_data: Optional[Dict[str, Any]] = None,
    topics: Optional[Dict[str, Any]] = None,
    participants: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a clean, human-readable meeting summary using Gemini.

    Returns a dict with at least:
      {
        "main_summary": str,
        "participant_count": int,
        "detected_names": List[str],
        "overall_sentiment": Dict,
        "key_topics": List[str],
        "llm_source": "gemini" | "fallback"
      }
    """

    print("Generating meeting summary (Gemini)...")

    transcript = (transcript or "").strip()
    MIN_WORDS = 20
    word_count = len(transcript.split())

    base_payload = {
        "participant_count": participants.get("count", 0) if participants else 0,
        "detected_names": participants.get("detected_names", [])
        if participants
        else [],
        "overall_sentiment": sentiment_data.get("overall_mood", {})
        if sentiment_data
        else {},
        "key_topics": [],
        "llm_source": "fallback",
    }

    if not transcript or word_count < MIN_WORDS:
        print(f"✗ Not enough transcript text for summarization (words={word_count})")
        return {
            **base_payload,
            "main_summary": "Not enough speech was detected to generate a meaningful summary.",
        }

    cleaned = _clean_transcript_for_summarization(transcript)
    max_chars = 12000
    cleaned = cleaned[:max_chars]

    if not _GEMINI_AVAILABLE:
        print(
            f"✗ Gemini SDK not available; "
            f"returning simple heuristic summary instead ({_GEMINI_IMPORT_ERROR})"
        )
        sentences = _split_into_sentences(cleaned)
        summary = " ".join(sentences[:5]) if sentences else cleaned[:500]
        return {
            **base_payload,
            "main_summary": summary,
            "llm_source": "fallback_sdk_missing",
        }

    try:
        summary_struct = _gemini_meeting_summary_call(
            cleaned, sentiment_data, topics, participants
        )
        enhanced = {
            **base_payload,
            "main_summary": summary_struct.get("main_summary", ""),
            "key_topics": summary_struct.get("key_topics", []),
            "llm_source": "gemini",
        }
        print("✓ Summary generated via Gemini")
        return enhanced
    except Exception as e:
        print(f"✗ Error generating summary via Gemini: {e}")
        sentences = _split_into_sentences(cleaned)
        fallback = " ".join(sentences[:5]) if sentences else cleaned[:500]
        return {
            **base_payload,
            "main_summary": fallback,
            "llm_source": "fallback_error",
        }


# -------------------------------------------------------------------
# PUBLIC API: extract_action_items
# -------------------------------------------------------------------

def extract_action_items(
    transcript: str,
    speaker_segments: Optional[List[Dict[str, Any]]] = None,
    participants: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract action items from transcript using Gemini.

    Output keeps the same shape as before (so the rest of your app
    doesn't need to change):

    [
      {
        "action": str,
        "role": "action" | "decision" | "blocker",
        "score": float,
        "confidence": "low" | "medium" | "high",
        "assignee": Optional[str],
        "speaker_id": Optional[str],
        "person_id": Optional[str],
        "start": Optional[float],
        "end": Optional[float],
      },
      ...
    ]
    """
    print("Extracting action items (Gemini)...")

    transcript = (transcript or "").strip()
    if not transcript:
        return []

    if not _GEMINI_AVAILABLE:
        print(
            f"✗ Gemini SDK not available; "
            f"returning no action items ({_GEMINI_IMPORT_ERROR})"
        )
        return []

    cleaned = _clean_transcript_for_summarization(transcript)
    max_chars = 12000
    cleaned = cleaned[:max_chars]

    try:
        gemini_items = _gemini_action_items_call(
            cleaned, speaker_segments=speaker_segments, participants=participants
        )
    except Exception as e:
        print(f"✗ Error extracting action items via Gemini: {e}")
        return []

    results: List[Dict[str, Any]] = []
    for item in gemini_items:
        action = item.get("action")
        if not action:
            continue

        role = item.get("role", "action")
        confidence = item.get("confidence", "medium")
        assignee = item.get("assignee")

        score = 0.5
        if confidence == "high":
            score = 0.9
        elif confidence == "medium":
            score = 0.7

        results.append(
            {
                "action": action,
                "role": role,
                "score": score,
                "confidence": confidence,
                "assignee": assignee,
                "speaker_id": None,  # we don't map back to diarization here
                "person_id": None,
                "start": None,
                "end": None,
            }
        )

    # Deduplicate
    seen = set()
    unique: List[Dict[str, Any]] = []
    for it in results:
        key = (it["action"].lower(), (it.get("assignee") or "").lower())
        if key not in seen:
            seen.add(key)
            unique.append(it)

    print(
        f"✓ Extracted {len(unique)} action items via Gemini"
        + (
            f" ({sum(1 for a in unique if a.get('assignee'))} with assignee)"
            if unique
            else ""
        )
    )
    return unique[:20]
