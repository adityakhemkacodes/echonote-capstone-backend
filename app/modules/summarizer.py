# app/modules/summarizer.py
"""
Gemini-based summarizer + action item extractor.

Updates:
- Post-process Gemini outputs to snap guessed names to detected participant names
  for THIS meeting (no hardcoded mapping).
"""

import os
import re
import json
import warnings
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)

warnings.filterwarnings("ignore")

_GEMINI_AVAILABLE = False
_GEMINI_IMPORT_ERROR = None

try:
    import google.generativeai as genai  # type: ignore
    _GEMINI_AVAILABLE = True
except Exception as e:  # pragma: no cover
    _GEMINI_AVAILABLE = False
    _GEMINI_IMPORT_ERROR = e


def _load_gemini_api_key() -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key.strip()

    try:
        here = Path(__file__).resolve()
        candidates = [
            here.parent / ".gemini_key",
            here.parent.parent / ".gemini_key",
            here.parent.parent.parent / ".gemini_key",
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
    if not _GEMINI_AVAILABLE:
        raise RuntimeError(f"Gemini SDK not available: {_GEMINI_IMPORT_ERROR}")

    api_key = _load_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key missing. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
            "or create a .gemini_key file with your API key."
        )

    genai.configure(api_key=api_key)

    raw_name = model_name or os.getenv("GEMINI_MODEL") or "models/gemini-2.5-flash"
    effective_name = (raw_name or "").strip()
    print(f"[Gemini] Using model: {effective_name}")
    return genai.GenerativeModel(effective_name)


_NOISE_KEYWORDS = [
    "coffee", "tea", "lunch", "snack", "pizza",
    "mascot", "cat ", "cats ", "joke", "laugh", "crying",
    "dark mode powered by chaos", "coffee machine",
    "construction noise", "wifi", "wi-fi", "headset",
]


def _split_into_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


def _is_noise_sentence(s: str) -> bool:
    lower = s.lower()
    return any(k in lower for k in _NOISE_KEYWORDS)


def _clean_transcript_for_summarization(text: str) -> str:
    sentences = _split_into_sentences(text)
    cleaned: List[str] = []

    for s in sentences:
        if len(s) < 15:
            continue

        lower = s.lower()
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

    if len(" ".join(cleaned).split()) < 30:
        return text.strip()

    return " ".join(cleaned)


def _safe_json_from_text(raw: str) -> Optional[Any]:
    raw = (raw or "").strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

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


def _extract_participant_names(participants: Optional[Union[Dict[str, Any], List[Any]]]) -> List[str]:
    if not participants:
        return []

    names: List[str] = []

    if isinstance(participants, list):
        for p in participants:
            if isinstance(p, dict):
                nm = p.get("canonical_name") or p.get("name") or p.get("display_name")
                if nm:
                    names.append(str(nm))
        out = []
        seen = set()
        for n in names:
            k = n.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(n.strip())
        return out

    if isinstance(participants, dict):
        direct = participants.get("detected_names") or participants.get("names") or []
        if isinstance(direct, list):
            for n in direct:
                if n:
                    names.append(str(n))

        plist = (
            participants.get("participants")
            or participants.get("participants_fused")
            or participants.get("speaker_registry")
            or []
        )
        if isinstance(plist, list):
            for p in plist:
                if not isinstance(p, dict):
                    continue
                nm = p.get("canonical_name") or p.get("name") or p.get("display_name")
                if nm:
                    names.append(str(nm))

    out = []
    seen = set()
    for n in names:
        k = n.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(n.strip())
    return out


def _extract_participant_count(participants: Optional[Union[Dict[str, Any], List[Any]]]) -> int:
    if not participants:
        return 0
    if isinstance(participants, list):
        return len(participants)
    if isinstance(participants, dict):
        c = participants.get("count") or participants.get("participant_count")
        if isinstance(c, int):
            return c
        names = _extract_participant_names(participants)
        return len(names)
    return 0


# ----------------------------
# NEW: name snapping helpers
# ----------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _name_similarity(a: str, b: str) -> float:
    """
    Token+char similarity (lightweight, no deps).
    """
    a_norm = _norm(a)
    b_norm = _norm(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens) or 1
    token_sim = inter / union

    # cheap char sim
    from difflib import SequenceMatcher
    char_sim = SequenceMatcher(None, a_norm, b_norm).ratio()

    return 0.6 * char_sim + 0.4 * token_sim


def _snap_name_to_participants(raw_name: Optional[str], participant_names: List[str]) -> Optional[str]:
    """
    Snap Gemini's guessed name to the closest detected participant name for THIS meeting.
    No manual mapping; just fuzzy match.
    """
    if not raw_name or not isinstance(raw_name, str):
        return None

    raw = raw_name.strip()
    if not raw:
        return None

    if not participant_names:
        return raw

    best = None
    best_score = 0.0

    for p in participant_names:
        sc = _name_similarity(raw, p)
        # slight boost if first token matches (helps "Adi Nader" -> "Aadi N")
        r0 = _norm(raw).split()[:1]
        p0 = _norm(p).split()[:1]
        if r0 and p0 and r0[0] == p0[0]:
            sc += 0.08
        if sc > best_score:
            best_score = sc
            best = p

    # conservative threshold: don't rewrite random stuff
    if best and best_score >= 0.62:
        return best
    return raw


def _replace_names_in_text(text: str, participant_names: List[str]) -> str:
    """
    Replace any 'near-miss' names in summary text with snapped names
    only when a clear match exists.
    """
    if not text or not participant_names:
        return text

    # Find candidate "Name-like" spans (2 tokens or token+initial).
    # We only replace when snapping changes it.
    pattern = re.compile(r"\b([A-Z][a-z]{1,20})(?:\s+([A-Z][a-z]{1,20}|[A-Z]))\b")
    def repl(m):
        cand = m.group(0)
        snapped = _snap_name_to_participants(cand, participant_names)
        return snapped if snapped and snapped != cand else cand

    return pattern.sub(repl, text)


def _gemini_meeting_summary_call(
    transcript: str,
    sentiment_data: Optional[Dict[str, Any]],
    topics: Optional[Dict[str, Any]],
    participants: Optional[Union[Dict[str, Any], List[Any]]],
) -> Dict[str, Any]:
    model = _get_gemini_model()

    meta_bits = []

    if sentiment_data and isinstance(sentiment_data, dict):
        overall = sentiment_data.get("overall_mood") or sentiment_data.get("overall") or {}
        if overall:
            meta_bits.append(f"Heuristic overall sentiment: {overall}")

    count = _extract_participant_count(participants)
    names = _extract_participant_names(participants)
    if count:
        meta_bits.append(f"Estimated participant count: {count}")
    if names:
        meta_bits.append("Detected participant names: " + ", ".join(names))

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
        main_summary = transcript[:500] + "..." if len(transcript) > 500 else transcript

    return {"main_summary": main_summary, "key_topics": key_topics}


def _gemini_action_items_call(
    transcript: str,
    speaker_segments: Optional[List[Dict[str, Any]]] = None,
    participants: Optional[Union[Dict[str, Any], List[Any]]] = None,
) -> List[Dict[str, Any]]:
    model = _get_gemini_model()

    participant_names = _extract_participant_names(participants)
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

        results.append({"action": action, "role": role, "confidence": confidence, "assignee": assignee})

    return results


def generate_summary(
    transcript: str,
    sentiment_data: Optional[Dict[str, Any]] = None,
    topics: Optional[Dict[str, Any]] = None,
    participants: Optional[Union[Dict[str, Any], List[Any]]] = None,
) -> Dict[str, Any]:
    print("Generating meeting summary (Gemini)...")

    transcript = (transcript or "").strip()
    MIN_WORDS = 20
    word_count = len(transcript.split())

    detected_names = _extract_participant_names(participants)
    participant_count = _extract_participant_count(participants)

    base_payload = {
        "participant_count": participant_count,
        "detected_names": detected_names,
        "overall_sentiment": sentiment_data.get("overall_mood", {}) if sentiment_data else {},
        "key_topics": [],
        "llm_source": "fallback",
    }

    if not transcript or word_count < MIN_WORDS:
        print(f"✗ Not enough transcript text for summarization (words={word_count})")
        return {**base_payload, "main_summary": "Not enough speech was detected to generate a meaningful summary."}

    cleaned = _clean_transcript_for_summarization(transcript)
    cleaned = cleaned[:12000]

    if not _GEMINI_AVAILABLE:
        print(f"✗ Gemini SDK not available; returning fallback ({_GEMINI_IMPORT_ERROR})")
        sentences = _split_into_sentences(cleaned)
        summary = " ".join(sentences[:5]) if sentences else cleaned[:500]
        return {**base_payload, "main_summary": summary, "llm_source": "fallback_sdk_missing"}

    try:
        summary_struct = _gemini_meeting_summary_call(cleaned, sentiment_data, topics, participants)

        main_summary = summary_struct.get("main_summary", "")
        main_summary = _replace_names_in_text(main_summary, detected_names)

        enhanced = {
            **base_payload,
            "main_summary": main_summary,
            "key_topics": summary_struct.get("key_topics", []),
            "llm_source": "gemini",
        }
        print("✓ Summary generated via Gemini")
        return enhanced
    except Exception as e:
        print(f"✗ Error generating summary via Gemini: {e}")
        sentences = _split_into_sentences(cleaned)
        fallback = " ".join(sentences[:5]) if sentences else cleaned[:500]
        return {**base_payload, "main_summary": fallback, "llm_source": "fallback_error"}


def extract_action_items(
    transcript: str,
    speaker_segments: Optional[List[Dict[str, Any]]] = None,
    participants: Optional[Union[Dict[str, Any], List[Any]]] = None,
) -> List[Dict[str, Any]]:
    print("Extracting action items (Gemini)...")

    transcript = (transcript or "").strip()
    if not transcript:
        return []

    if not _GEMINI_AVAILABLE:
        print(f"✗ Gemini SDK not available; returning no action items ({_GEMINI_IMPORT_ERROR})")
        return []

    cleaned = _clean_transcript_for_summarization(transcript)
    cleaned = cleaned[:12000]

    participant_names = _extract_participant_names(participants)

    try:
        gemini_items = _gemini_action_items_call(cleaned, speaker_segments=speaker_segments, participants=participants)
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
        if assignee:
            assignee = _snap_name_to_participants(str(assignee), participant_names)

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
                "speaker_id": None,
                "person_id": None,
                "start": None,
                "end": None,
            }
        )

    seen = set()
    unique: List[Dict[str, Any]] = []
    for it in results:
        key = (it["action"].lower(), (it.get("assignee") or "").lower())
        if key not in seen:
            seen.add(key)
            unique.append(it)

    print(
        f"✓ Extracted {len(unique)} action items via Gemini"
        + (f" ({sum(1 for a in unique if a.get('assignee'))} with assignee)" if unique else "")
    )
    return unique[:20]
