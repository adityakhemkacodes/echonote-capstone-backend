# app/modules/summarizer.py

import re
import warnings
from typing import List, Dict, Any, Optional

from transformers import pipeline

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Global models
# -------------------------------------------------------------------

_summarizer = None
_zsc = None  # zero-shot classifier

# Sentences containing any of these keywords will be dropped
# from summary + action items as "chit-chat".
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
    "dark mode",
    "coffee machine",
    "weather",
    "weekend",
]


def _get_summarizer():
    """Lazy-load transformer summarization model."""
    global _summarizer
    if _summarizer is None:
        print("Loading transformer summarization model (bart-large-cnn)...")
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
        )
    return _summarizer


def _get_zsc():
    """Lazy-load zero-shot classification model."""
    global _zsc
    if _zsc is None:
        print("Loading zero-shot classifier (bart-large-mnli)...")
        _zsc = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _zsc


# -------------------------------------------------------------------
# Basic text helpers
# -------------------------------------------------------------------

def _split_into_sentences(text: str) -> List[str]:
    """Crude sentence splitter for meeting transcripts."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 0]


def _is_noise_sentence(s: str) -> bool:
    lower = s.lower()
    if len(lower.split()) < 4:
        # very short sentences are usually not helpful
        return True
    return any(k in lower for k in _NOISE_KEYWORDS)


def _clean_sentence(s: str) -> Optional[str]:
    """Filter obviously broken / useless sentences."""
    s = (s or "").strip()
    if not s:
        return None
    if len(s) < 25:  # very tiny fragments
        return None
    if len(s) > 400:  # way too long
        return None
    if _is_noise_sentence(s):
        return None
    return s


def _summarize_long_text(text: str, max_chunk_chars: int = 2500) -> str:
    """
    Summarize arbitrarily long text by:
      1) chunking
      2) summarizing each chunk
      3) summarizing the concatenated chunk summaries
    """
    summarizer = _get_summarizer()

    text = text.strip()
    if len(text) <= max_chunk_chars:
        result = summarizer(
            text,
            max_length=220,
            min_length=60,
            do_sample=False,
        )[0]["summary_text"]
        return result.strip()

    # Split into overlapping chunks
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_chars
        chunk = text[start:end]
        # try not to cut in the middle of a sentence
        last_period = chunk.rfind(".")
        if last_period > 0 and last_period > len(chunk) * 0.6:
            chunk = chunk[: last_period + 1]
            end = start + last_period + 1
        chunks.append(chunk)
        start = end

    partial_summaries = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        res = summarizer(
            ch,
            max_length=160,
            min_length=40,
            do_sample=False,
        )[0]["summary_text"]
        partial_summaries.append(res.strip())

    joined = " ".join(partial_summaries)
    final = summarizer(
        joined,
        max_length=230,
        min_length=70,
        do_sample=False,
    )[0]["summary_text"]
    return final.strip()


# -------------------------------------------------------------------
# Transformer-augmented summary
# -------------------------------------------------------------------

def generate_summary(
    transcript: str,
    sentiment_data: Dict[str, Any] = None,
    topics: Dict[str, Any] = None,
    participants: Dict[str, Any] = None,
):
    """
    Generate comprehensive meeting summary, focusing on:
      - high-level transformer-based overview
      - key discussion points (from topic segmentation)
      - decisions
      - action highlights

    Returns a dict with:
      {
        "main_summary": str,          # human-readable block (for CLI)
        "overview": str,              # transformer summary paragraph
        "key_points": List[str],
        "decisions": List[str],
        "action_highlights": List[str],
        "participant_count": int,
        "detected_names": List[str],
        "overall_sentiment": Dict,
        "key_topics": List[Any],
      }
    """
    print("Generating meeting summary...")

    transcript = (transcript or "").strip()
    MIN_WORDS = 20
    word_count = len(transcript.split())

    base_payload = {
        "participant_count": participants.get("count", 0) if participants else 0,
        "detected_names": participants.get("detected_names", []) if participants else [],
        "overall_sentiment": sentiment_data.get("overall_mood", {}) if sentiment_data else {},
        "key_topics": topics.get("topics", [])[:3] if topics and isinstance(topics, dict) else [],
    }

    if not transcript or word_count < MIN_WORDS:
        print(f"✗ Not enough transcript text for summarization (words={word_count})")
        return {
            **base_payload,
            "main_summary": "Not enough speech was detected to generate a meaningful summary.",
            "overview": "",
            "key_points": [],
            "decisions": [],
            "action_highlights": [],
        }

    # Cut extremely long transcripts to keep things bounded
    max_chars = 12000
    transcript = transcript[:max_chars]

    try:
        # 1) High-level overview with transformer summarizer
        overview = _summarize_long_text(transcript)

        # 2) Key discussion points – lean on topic segmentation
        key_points: List[str] = []
        if topics and isinstance(topics, dict) and topics.get("topics"):
            for t in topics["topics"][:3]:
                summary = (t.get("summary") or "").strip()
                if summary:
                    key_points.append(summary)

        # Fallback if no topics or no summaries
        if not key_points:
            sentences = [_clean_sentence(s) for s in _split_into_sentences(transcript)]
            key_points = [s for s in sentences if s][:3]

        # 3) Decisions & action highlights via zero-shot classifier
        zsc = _get_zsc()
        sentences = _split_into_sentences(transcript)
        # keep only reasonably sized, non-noise sentences
        sentences = [s for s in (_clean_sentence(s) for s in sentences) if s]

        # To control cost, limit to first N sentences
        MAX_SENTENCES_FOR_CLASS = 80
        sentences = sentences[:MAX_SENTENCES_FOR_CLASS]

        decisions: List[str] = []
        action_highlights: List[str] = []

        candidate_labels = ["decision", "action item", "status update", "small talk"]

        for s in sentences:
            result = zsc(s, candidate_labels=candidate_labels, multi_label=False)
            if not result or not result.get("labels"):
                continue
            label = result["labels"][0]
            score = float(result["scores"][0])

            if label == "decision" and score >= 0.8:
                if s not in decisions:
                    decisions.append(s)
            elif label == "action item" and score >= 0.8:
                if s not in action_highlights:
                    action_highlights.append(s)

            # stop if we have enough
            if len(decisions) >= 5 and len(action_highlights) >= 5:
                break

        # 4) Build human-readable block summary for CLI
        lines: List[str] = []
        if overview:
            lines.append(overview.strip())
            lines.append("")

        if key_points:
            lines.append("Key discussion points:")
            for s in key_points:
                lines.append(f"- {s}")
            lines.append("")

        if decisions:
            lines.append("Decisions made:")
            for s in decisions:
                lines.append(f"- {s}")
            lines.append("")

        if action_highlights:
            lines.append("Action highlights:")
            for s in action_highlights[:5]:
                lines.append(f"- {s}")
            lines.append("")

        if not lines:
            # extreme fallback – shouldn't really happen
            lines = [overview or transcript[:500]]

        main_summary_text = "\n".join(lines).strip()

        enhanced_summary = {
            **base_payload,
            "main_summary": main_summary_text,
            "overview": overview,
            "key_points": key_points,
            "decisions": decisions,
            "action_highlights": action_highlights,
        }

        print("✓ Summary generated")
        return enhanced_summary

    except Exception as e:
        print(f"✗ Error generating summary: {e}")
        fallback = transcript[:500] + "..." if len(transcript) > 500 else transcript
        return {
            **base_payload,
            "main_summary": fallback,
            "overview": "",
            "key_points": [],
            "decisions": [],
            "action_highlights": [],
        }


# -------------------------------------------------------------------
# Action items extraction (regex + zero-shot filtering)
# -------------------------------------------------------------------

def extract_action_items(
    transcript: str,
    speaker_segments: Optional[List[Dict[str, Any]]] = None,
    participants: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract action items and tasks from meeting transcript.

    Pipeline:
      1) regex to find candidate action phrases
      2) clean up fragments
      3) use zero-shot classifier to keep only real "action item" sentences
      4) simple dedupe

    Returns a list of dicts:
      { "action": str, "confidence": "medium|high", "assignee": Optional[str] }
    """
    print("Extracting action items...")

    transcript = (transcript or "").strip()
    if not transcript:
        return []

    # --- Step 1: regex-based candidates ---
    raw_candidates: List[str] = []

    action_patterns = [
        r"(?:will|should|need to|must|have to|going to)\s+([^.!?]+)",
        r"action item[s]?:\s*([^.!?]+)",
        r"to-?do[s]?:\s*([^.!?]+)",
        r"(?:please|let's)\s+([^.!?]+)",
        r"(?:I|we|you|they)(?:'ll| will)\s+([^.!?]+)",
    ]

    for pattern in action_patterns:
        matches = re.finditer(pattern, transcript, re.IGNORECASE)
        for match in matches:
            fragment = match.group(1).strip()
            if not fragment:
                continue

            # Trim at first sentence boundary
            fragment = re.split(r"[.;]", fragment)[0].strip()

            # Drop very short / very long
            if not (10 < len(fragment) < 200):
                continue

            lower = fragment.lower()
            if any(k in lower for k in _NOISE_KEYWORDS):
                continue

            # Require at least one "work-like" verb
            if not re.search(
                r"\b("
                r"build|fix|finish|implement|test|deploy|create|update|review|prepare|"
                r"send|schedule|refactor|write|design|discuss|analyze|monitor|complete|done"
                r")\b",
                lower,
            ):
                # still keep some generic phrases like "be done before Friday"
                if not re.search(r"\b(done|complete|before)\b", lower):
                    continue

            raw_candidates.append(fragment)

    if not raw_candidates:
        print("✓ No regex-based action candidates found")
        return []

    # --- Step 2: zero-shot classification to keep genuine actions ---
    zsc = _get_zsc()
    labels = ["action item", "decision", "status update", "small talk"]

    filtered_items: List[Dict[str, Any]] = []

    for frag in raw_candidates:
        try:
            result = zsc(frag, candidate_labels=labels, multi_label=False)
            if not result or not result.get("labels"):
                continue
            top_label = result["labels"][0]
            score = float(result["scores"][0])

            if top_label != "action item" or score < 0.75:
                continue

            confidence = "high" if score >= 0.9 else "medium"
            filtered_items.append(
                {
                    "action": frag,
                    "confidence": confidence,
                    "assignee": None,  # ready for future identity linking
                }
            )
        except Exception:
            # If classifier breaks on a sentence, just skip it
            continue

    # --- Step 3: dedupe by normalized text ---
    seen = set()# app/modules/summarizer.py

import re
import warnings
from typing import List, Dict, Any, Optional

from transformers import pipeline

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Global models (lazy loaded)
# -------------------------------------------------------------------

_summarizer = None
_zs_classifier = None

# Sentences containing any of these keywords will be dropped
# from summarization and action extraction as "chit-chat".
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


def _get_summarizer():
    """Lazy-load abstractive summarization model."""
    global _summarizer
    if _summarizer is None:
        print("Loading transformer summarization model (bart-large-cnn)...")
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer


def _get_zs_classifier():
    """Lazy-load zero-shot classifier."""
    global _zs_classifier
    if _zs_classifier is None:
        print("Loading zero-shot classifier (bart-large-mnli)...")
        _zs_classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )
    return _zs_classifier


# -------------------------------------------------------------------
# Basic text utilities
# -------------------------------------------------------------------


def _split_into_sentences(text: str) -> List[str]:
    """Crude sentence splitter, good enough for transcripts."""
    text = (text or "").strip()
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if len(s.strip()) > 0]


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


# -------------------------------------------------------------------
# Main summary generation
# -------------------------------------------------------------------


def generate_summary(
    transcript: str,
    sentiment_data: Dict[str, Any] = None,
    topics: Dict[str, Any] = None,
    participants: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Generate a clean, human-readable meeting summary.

    - Uses BART (facebook/bart-large-cnn) for abstractive summarization
    - Cleans out obvious noise before summarizing
    - Optionally uses topic segments for "key topics"
    """

    print("Generating meeting summary...")

    transcript = (transcript or "").strip()
    MIN_WORDS = 20
    word_count = len(transcript.split())

    if not transcript or word_count < MIN_WORDS:
        print(f"✗ Not enough transcript text for summarization (words={word_count})")
        return {
            "main_summary": "Not enough speech was detected to generate a meaningful summary.",
            "participant_count": participants.get("count", 0) if participants else 0,
            "detected_names": participants.get("detected_names", [])
            if participants
            else [],
            "overall_sentiment": sentiment_data.get("overall_mood", {})
            if sentiment_data
            else {},
            "key_topics": topics.get("topics", [])[:3] if topics else [],
        }

    try:
        # 1) Clean transcript (remove chit-chat, noise etc.)
        cleaned = _clean_transcript_for_summarization(transcript)

        # Limit to ~8000 characters for the summarization model
        max_chars = 8000
        cleaned = cleaned[:max_chars]

        summarizer = _get_summarizer()

        # 2) High-level abstractive summary
        # Adjust max/min length according to your typical meeting size
        summary_output = summarizer(
            cleaned,
            max_length=180,
            min_length=60,
            do_sample=False,
            truncation=True,
        )[0]["summary_text"].strip()

        # 3) Optional: derive concise key topic summaries
        key_topics: List[str] = []
        if isinstance(topics, dict) and topics.get("topics"):
            # Take top 2–3 topic blobs and compress each a bit
            for t in topics["topics"][:3]:
                txt = (t.get("summary") or "").strip()
                if not txt:
                    continue
                # Short summarization per topic (keep it cheap)
                try:
                    topic_sum = summarizer(
                        txt,
                        max_length=60,
                        min_length=20,
                        do_sample=False,
                        truncation=True,
                    )[0]["summary_text"].strip()
                    key_topics.append(topic_sum)
                except Exception:
                    key_topics.append(txt[:200])

        enhanced_summary = {
            "main_summary": summary_output,
            "participant_count": participants.get("count", 0) if participants else 0,
            "detected_names": participants.get("detected_names", [])
            if participants
            else [],
            "overall_sentiment": sentiment_data.get("overall_mood", {})
            if sentiment_data
            else {},
            "key_topics": key_topics,
        }

        print("✓ Summary generated")
        return enhanced_summary

    except Exception as e:
        print(f"✗ Error generating summary: {e}")
        fallback = transcript[:500] + "..." if len(transcript) > 500 else transcript
        return {
            "main_summary": fallback,
            "participant_count": participants.get("count", 0) if participants else 0,
            "detected_names": participants.get("detected_names", [])
            if participants
            else [],
            "overall_sentiment": sentiment_data.get("overall_mood", {})
            if sentiment_data
            else {},
            "key_topics": topics.get("topics", [])[:3] if topics else [],
        }


# -------------------------------------------------------------------
# Action items extraction (speaker-aware + zero-shot + regex)
# -------------------------------------------------------------------

def _has_action_cue(text: str) -> bool:
    """Cheap filter to pre-select sentences that might contain an action."""
    lower = text.lower()
    cues = [
        "i'll ",
        "i will ",
        "we'll ",
        "we will ",
        "you will ",
        "you'll ",
        "should ",
        "need to ",
        "have to ",
        "must ",
        "going to ",
        "let's ",
        "to do ",
        "todo ",
        "action item",
        "before next meeting",
        "before friday",
        "by friday",
        "finish",
        "complete",
        "fix",
        "implement",
        "test ",
        "deploy",
        "update",
        "write ",
        "create ",
        "prepare ",
        "schedule ",
        "regain access",
    ]
    return any(c in lower for c in cues)


def _regex_extract_from_sentence(s: str) -> str:
    """
    Try to slice out the 'action' part from a sentence using regex.
    If nothing matches, returns the original sentence trimmed.
    """
    patterns = [
        r"(?:I|we|you|they)(?:'ll| will)\s+([^.!?]+)",
        r"(?:will|should|need to|must|have to|going to)\s+([^.!?]+)",
        r"(?:please|let's)\s+([^.!?]+)",
        r"action item[s]?:\s*([^.!?]+)",
        r"to-?do[s]?:\s*([^.!?]+)",
    ]

    for pat in patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return s.strip()


def _normalize_action_text(text: str) -> str:
    """
    Light cleanup to make action strings more readable, without
    trying to fully rewrite them (no LLM in the loop here).
    """
    s = (text or "").strip()

    # Drop leading filler like "and lastly", "okay", etc.
    s = re.sub(
        r"^(and\s+lastly|and\s+also|and|so|then|okay|ok|fine|lastly|also|just)\s+",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()

    # Normalise contractions we see a lot
    s = re.sub(r"\bI'll\b", "I will", s, flags=re.IGNORECASE)

    # Ensure it starts with a capital letter
    if s and not s[0].isupper():
        s = s[0].upper() + s[1:]

    # Add period if missing, for nicer display
    if s and s[-1] not in ".!?":
        s = s + "."

    return s


def _speaker_and_person_for_segment(
    seg: Dict[str, Any], participants: Optional[Dict[str, Any]]
) -> (Optional[str], Optional[str], Optional[str]):
    """
    Resolve speaker_id -> (person_id, canonical name) where possible.
    Returns (speaker_id, person_id, name).
    """
    sid = seg.get("speaker_id") or seg.get("speaker")
    if not participants or not sid:
        return sid, None, None

    speaker_to_person = participants.get("speaker_to_person", {}) or {}
    person_to_name = participants.get("person_to_name", {}) or {}

    participants_list = (
        participants.get("participants")
        or participants.get("participants_fused")
        or []
    )
    pid_to_canonical = {
        p.get("participant_id"): p.get("canonical_name")
        for p in participants_list
        if p.get("participant_id")
    }

    pid = speaker_to_person.get(sid)
    if not pid:
        return sid, None, None

    name = person_to_name.get(pid) or pid_to_canonical.get(pid)
    return sid, pid, name


def _extract_actions_speaker_aware(
    transcript: str,
    speaker_segments: Optional[List[Dict[str, Any]]],
    participants: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract action items using diarized speaker segments + zero-shot classification.
    This is the 'smart' path that can attach assignees reliably.
    """
    if not speaker_segments:
        return []

    classifier = _get_zs_classifier()

    # 1) Collect candidate segments that *might* contain actions
    candidate_segments: List[Dict[str, Any]] = []
    candidate_texts: List[str] = []

    for seg in speaker_segments:
        text = (seg.get("text") or "").strip()
        if len(text) < 20:
            continue
        if _is_noise_sentence(text):
            continue
        if not _has_action_cue(text):
            continue

        candidate_segments.append(seg)
        candidate_texts.append(text)

    if not candidate_segments:
        return []

    # 2) Run zero-shot on batch of candidate segment texts
    labels = ["action", "decision", "blocker", "other"]
    zs_results = classifier(
        candidate_texts, candidate_labels=labels, multi_label=False
    )
    if isinstance(zs_results, dict):
        zs_results = [zs_results]

    action_items: List[Dict[str, Any]] = []
    seen = set()

    for seg, res in zip(candidate_segments, zs_results):
        top_label = res["labels"][0]
        top_score = float(res["scores"][0])

        # Keep strong "action" / "blocker" / "decision" items
        if top_label not in ("action", "decision", "blocker") or top_score < 0.6:
            continue

        full_text = (seg.get("text") or "").strip()
        raw_action = _regex_extract_from_sentence(full_text)
        if len(raw_action) < 10:
            continue
        if _is_noise_sentence(raw_action):
            continue

        normalized_action = _normalize_action_text(raw_action)

        sid, pid, assignee_name = _speaker_and_person_for_segment(seg, participants)

        key = (normalized_action.lower(), assignee_name.lower() if assignee_name else "")
        if key in seen:
            continue
        seen.add(key)

        action_items.append(
            {
                "action": normalized_action,
                "role": top_label,                  # "action" | "decision" | "blocker"
                "score": round(top_score, 3),
                "confidence": "high" if top_score >= 0.85 else "medium",
                "assignee": assignee_name,
                "speaker_id": sid,
                "person_id": pid,
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
            }
        )

    return action_items


def _extract_actions_regex_only(transcript: str) -> List[Dict[str, Any]]:
    """
    Fallback: regex-based action extraction with no speaker/participant info.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return []

    action_items: List[Dict[str, Any]] = []

    patterns = [
        r"(?:will|should|need to|must|have to|going to)\s+([^.!?]+)",
        r"action item[s]?:\s*([^.!?]+)",
        r"to-?do[s]?:\s*([^.!?]+)",
        r"(?:please|let's)\s+([^.!?]+)",
        r"(?:I|we|you|they)(?:'ll| will)\s+([^.!?]+)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, transcript, re.IGNORECASE):
            raw = match.group(1).strip()
            if not (10 < len(raw) < 200):
                continue
            if _is_noise_sentence(raw):
                continue

            normalized_action = _normalize_action_text(raw)
            lower = normalized_action.lower()

            # Require at least one "work-like" verb
            if not re.search(
                r"\b("
                r"build|fix|finish|implement|test|deploy|create|update|review|prepare|"
                r"send|schedule|refactor|write|design|discuss|analyze|monitor|complete|"
                r"learn|study|improve|optimize"
                r")\b",
                lower,
            ):
                if not re.search(r"\b(done|complete|before)\b", lower):
                    continue

            action_items.append(
                {
                    "action": normalized_action,
                    "role": "action",
                    "score": 0.5,
                    "confidence": "medium",
                    "assignee": None,
                    "speaker_id": None,
                    "person_id": None,
                    "start": None,
                    "end": None,
                }
            )

    # Deduplicate
    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in action_items:
        key = (item["action"].lower(), item.get("assignee") or "")
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def extract_action_items(
    transcript: str,
    speaker_segments: Optional[List[Dict[str, Any]]] = None,
    participants: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Public entrypoint used by MeetingProcessor.

    - Prefer speaker-aware extraction (diarization + zero-shot).
    - Fall back to regex-only if we don't have diarized segments.
    - Always return deduplicated, normalized action items with
      assignee info when possible.
    """
    print("Extracting action items...")

    transcript = (transcript or "").strip()
    if not transcript:
        return []

    items: List[Dict[str, Any]] = []

    # Preferred path: diarized segments + participants
    if speaker_segments:
        items = _extract_actions_speaker_aware(transcript, speaker_segments, participants)

    # Fallback: global regex if nothing turned up
    if not items:
        items = _extract_actions_regex_only(transcript)

    # Final safety dedupe
    seen = set()
    unique: List[Dict[str, Any]] = []
    for it in items:
        key = (
            (it.get("action") or "").lower(),
            it.get("assignee") or "",
            it.get("speaker_id") or "",
        )
        if key not in seen:
            seen.add(key)
            unique.append(it)

    print(
        f"✓ Extracted {len(unique)} action items"
        + (
            f" ({sum(1 for a in unique if a.get('assignee'))} with assignee)"
            if unique
            else ""
        )
    )
    return unique[:15]