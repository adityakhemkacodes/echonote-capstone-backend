# app/modules/summarizer.py
import re
import warnings
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")

_embedder = None
_proto_built = False
_action_proto = None
_decision_proto = None
_context_proto = None

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
]

# Prototype phrases for semantic classification
_ACTION_PROTOTYPES = [
    "finish the feature",
    "implement the API",
    "fix the bug",
    "deploy the change",
    "write tests",
    "update the documentation",
    "schedule a follow up meeting",
    "create a ticket",
    "prepare the report",
    "complete the task",
]

_DECISION_PROTOTYPES = [
    "we decided that",
    "we will go with this approach",
    "we agreed to",
    "the final decision is",
    "we chose this option",
    "we will not implement this",
    "we will postpone this",
]

_CONTEXT_PROTOTYPES = [
    "today's meeting is about",
    "current status of the project",
    "progress update",
    "overview of the sprint",
    "discussion about the roadmap",
    "summary of the work so far",
]


def _get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading sentence embedding model (all-MiniLM-L6-v2)...")
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def _build_prototypes():
    global _proto_built, _action_proto, _decision_proto, _context_proto
    if _proto_built:
        return

    print("Building semantic class prototypes for sentence classification...")
    model = _get_embedder()

    action_emb = model.encode(_ACTION_PROTOTYPES, convert_to_tensor=True)
    decision_emb = model.encode(_DECISION_PROTOTYPES, convert_to_tensor=True)
    context_emb = model.encode(_CONTEXT_PROTOTYPES, convert_to_tensor=True)

    # Average each class into a single prototype vector
    _action_proto = action_emb.mean(dim=0, keepdim=True)
    _decision_proto = decision_emb.mean(dim=0, keepdim=True)
    _context_proto = context_emb.mean(dim=0, keepdim=True)
    _proto_built = True


def _split_into_sentences(text: str) -> List[str]:
    # crude but works well enough for meeting transcripts
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 0]


def _is_noise_sentence(s: str) -> bool:
    lower = s.lower()
    return any(k in lower for k in _NOISE_KEYWORDS)


def _classify_sentences(sentences: List[str]) -> List[Dict[str, Any]]:
    """
    For each sentence, compute similarity to action/decision/context prototypes.
    Returns list of dicts with classification info.
    """
    if not sentences:
        return []

    _build_prototypes()
    model = _get_embedder()

    sent_embs = model.encode(sentences, convert_to_tensor=True)

    action_sim = util.cos_sim(sent_embs, _action_proto)[:, 0]
    decision_sim = util.cos_sim(sent_embs, _decision_proto)[:, 0]
    context_sim = util.cos_sim(sent_embs, _context_proto)[:, 0]

    results = []
    for idx, s in enumerate(sentences):
        a = float(action_sim[idx])
        d = float(decision_sim[idx])
        c = float(context_sim[idx])

        # choose label by max similarity
        label = "context"
        score = c
        if a > d and a > c:
            label = "action"
            score = a
        elif d > a and d > c:
            label = "decision"
            score = d

        results.append(
            {
                "sentence": s,
                "label": label,
                "score": score,
                "action_score": a,
                "decision_score": d,
                "context_score": c,
            }
        )

    return results


def _select_top_sentences(classified, label, max_items, min_score=0.2):
    """
    Pick top sentences of a given label, with score threshold
    and noise filtering.
    """
    candidates = [
        c for c in classified if c["label"] == label and c["score"] >= min_score
    ]
    # keep original order by transcript, but bias to higher scores
    # sort by score desc but keep stable ordering via enumerate index
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda x: (-x[1]["score"], x[0]))
    chosen = []

    for _, c in indexed:
        s = c["sentence"].strip()
        if _is_noise_sentence(s):
            continue
        if s not in chosen:
            chosen.append(s)
        if len(chosen) >= max_items:
            break

    return chosen


def generate_summary(
    transcript: str,
    sentiment_data: Dict[str, Any] = None,
    topics: Dict[str, Any] = None,
    participants: Dict[str, Any] = None,
):
    """
    Generate comprehensive meeting summary, focusing on:
    - main discussion points
    - decisions
    - action items

    Uses sentence-transformers to classify sentences instead of
    a generative LLM, so it runs fully locally.
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
            "detected_names": participants.get("detected_names", []) if participants else [],
            "overall_sentiment": sentiment_data.get("overall_mood", {}) if sentiment_data else {},
            "key_topics": topics.get("topics", [])[:3] if topics else [],
        }

    # Cut extremely long transcripts
    max_chars = 8000
    transcript = transcript[:max_chars]

    try:
        sentences = _split_into_sentences(transcript)

        # Remove ultra-short or ultra-long sentences
        sentences = [
            s for s in sentences if 10 <= len(s) <= 300
        ]  # filter extremes

        classified = _classify_sentences(sentences)

        # Key points: mostly context + some high-scoring actions/decisions
        key_context = _select_top_sentences(classified, "context", max_items=5, min_score=0.15)
        key_decisions = _select_top_sentences(classified, "decision", max_items=4, min_score=0.20)
        key_actions = _select_top_sentences(classified, "action", max_items=6, min_score=0.20)

        # If no action sentences were classified, we still fall back to regex-based extraction
        # via extract_action_items() for the "Action items" section below.

        # Build human-readable block summary
        lines = []

        if key_context:
            lines.append("Key discussion points:")
            for s in key_context:
                lines.append(f"- {s}")
            lines.append("")

        if key_decisions:
            lines.append("Decisions made:")
            for s in key_decisions:
                lines.append(f"- {s}")
            lines.append("")

        # Note: the actual action items list is generated below and printed in run_local_test;
        # here we only provide a brief subset in the summary text.
        if key_actions:
            lines.append("Action highlights:")
            for s in key_actions[:3]:
                lines.append(f"- {s}")
            lines.append("")

        # If we somehow got nothing, fall back to first 3 sentences
        if not lines:
            fallback = " ".join(sentences[:3])
            lines = [fallback]

        main_summary_text = "\n".join(lines).strip()

        enhanced_summary = {
            "main_summary": main_summary_text,
            "participant_count": participants.get("count", 0) if participants else 0,
            "detected_names": participants.get("detected_names", []) if participants else [],
            "overall_sentiment": sentiment_data.get("overall_mood", {}) if sentiment_data else {},
            "key_topics": topics.get("topics", [])[:3] if topics else [],
        }

        print("✓ Summary generated")
        return enhanced_summary

    except Exception as e:
        print(f"✗ Error generating summary: {e}")
        fallback = transcript[:500] + "..." if len(transcript) > 500 else transcript
        return {
            "main_summary": fallback,
            "participant_count": participants.get("count", 0) if participants else 0,
            "detected_names": participants.get("detected_names", []) if participants else [],
            "overall_sentiment": sentiment_data.get("overall_mood", {}) if sentiment_data else {},
            "key_topics": topics.get("topics", [])[:3] if topics else [],
        }


# -------------------------------------------------------------------
# Action items extraction (regex + noise filtering)
# -------------------------------------------------------------------

def extract_action_items(
    transcript: str,
    speaker_segments: Optional[List[Dict[str, Any]]] = None,
    participants: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract action items and tasks from meeting transcript.

    - Uses regex pattern matching to find action-oriented phrases.
    - Filters out obvious non-business chatter (coffee, jokes, etc).
    - If identity information is available (participants_fused / speaker map),
      this function is ready to add assignee information later, but for now
      it keeps results text-only and backward compatible.

    Returns a list of dicts:
      { "action": str, "confidence": "medium", "assignee": Optional[str] }
    """
    print("Extracting action items...")

    transcript = (transcript or "").strip()
    if not transcript:
        return []

    action_items: List[Dict[str, Any]] = []

    # Common action phrases
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
            action_text = match.group(1).strip()

            # Filter out very short or very long items
            if not (10 < len(action_text) < 200):
                continue

            lower = action_text.lower()

            # Drop noisy / non-business actions
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

            # --- Assignee placeholder (future-proof for identity fusion) ---
            assignee: Optional[str] = None

            # NOTE: To reliably map this action to a specific speaker/name,
            # we would need word- or sentence-level timestamps from Whisper
            # and align them with diarization + participants_fused.
            # For now we keep 'assignee' = None to avoid incorrect attribution.

            action_items.append(
                {
                    "action": action_text,
                    "confidence": "medium",
                    "assignee": assignee,
                }
            )

    # Remove duplicates (case-insensitive on action text)
    seen = set()
    unique_actions: List[Dict[str, Any]] = []
    for item in action_items:
        key = item["action"].lower()
        if key not in seen:
            seen.add(key)
            unique_actions.append(item)

    result = unique_actions[:15]
    print(f"✓ Extracted {len(result)} action items")
    return result
