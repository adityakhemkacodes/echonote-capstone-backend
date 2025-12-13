# app/modules/entity_recognition.py

import warnings
from typing import List, Dict, Optional, Set, Tuple

warnings.filterwarnings("ignore")

try:
    import spacy  # type: ignore
except Exception:
    spacy = None

_nlp = None


def get_nlp():
    """
    Lazy load spaCy model.
    Returns:
      - nlp object, or
      - None if unavailable
    """
    global _nlp
    if _nlp is not None:
        return _nlp

    if spacy is None:
        print("⚠ spaCy not installed. Entity extraction disabled.")
        _nlp = None
        return _nlp

    try:
        print("Loading spaCy model (en_core_web_sm)...")
        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except Exception:
        print("⚠ spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        _nlp = None
        return _nlp


def _iter_chunks(text: str, chunk_chars: int) -> List[str]:
    """
    Chunk text into roughly chunk_chars segments, attempting to split on sentence-ish boundaries.
    """
    if not text:
        return []
    chunk_chars = max(2000, int(chunk_chars or 12000))

    chunks: List[str] = []
    n = len(text)
    i = 0

    while i < n:
        j = min(n, i + chunk_chars)

        # Try to extend to a nicer boundary (newline / sentence end) within a small window
        window_end = min(n, j + 600)
        window = text[j:window_end]

        cut = None
        for sep in ["\n\n", "\n", ". ", "? ", "! "]:
            k = window.find(sep)
            if k != -1:
                cut = j + k + len(sep)
                break

        if cut is not None and cut > i:
            j = cut

        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)

        i = j

    return chunks


def extract_entities(
    text: str,
    max_chars: int = 60000,
    chunk_chars: int = 12000,
    allowed_labels: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Extract named entities using spaCy NER.

    Output: [{"text": "...", "label": "PERSON"}, ...]

    Notes:
    - Robust for long transcripts (chunking).
    - Uses nlp.pipe for speed.
    - De-dupes case-insensitively on (text,label).
    """
    print("Extracting named entities...")

    text = (text or "").strip()
    if not text:
        print("✗ Entity extraction skipped (empty text)")
        return []

    nlp = get_nlp()
    if nlp is None:
        print("✗ Entity extraction skipped (spaCy not available)")
        return []

    if len(text) > int(max_chars or 60000):
        text = text[: int(max_chars or 60000)]

    chunks = _iter_chunks(text, chunk_chars=int(chunk_chars or 12000))
    if not chunks:
        return []

    # Default: keep the common “useful” ones for meetings
    if allowed_labels is None:
        allowed_labels = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "DATE", "TIME", "MONEY"}

    entities: List[Dict] = []
    seen: Set[Tuple[str, str]] = set()

    try:
        # Disable components we don't need for NER; keep NER enabled
        disable = ["tagger", "parser", "attribute_ruler", "lemmatizer"]
        # nlp.pipe is much faster than calling nlp() per chunk
        with nlp.select_pipes(disable=[p for p in disable if p in nlp.pipe_names]):
            for doc in nlp.pipe(chunks, batch_size=8):
                for ent in doc.ents:
                    label = (ent.label_ or "").strip()
                    if allowed_labels and label not in allowed_labels:
                        continue

                    ent_text = (ent.text or "").strip()
                    if not ent_text:
                        continue

                    key = (ent_text.lower(), label)
                    if key in seen:
                        continue
                    seen.add(key)

                    entities.append({"text": ent_text, "label": label})

        print(f"✓ Extracted {len(entities)} entities")
        return entities

    except Exception as e:
        print(f"✗ Entity extraction error: {e}")
        return []
