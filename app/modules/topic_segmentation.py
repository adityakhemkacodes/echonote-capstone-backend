# app/modules/topic_segmentation.py

from typing import Any, Dict, List, Optional, Tuple
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def _clean_chunk(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _strip_speaker_prefix(text: str) -> str:
    """
    If transcript lines look like "Aditya: blah blah", remove "Aditya:".
    This improves TF-IDF quality.
    """
    t = (text or "").strip()
    # "Name: text"
    t = re.sub(r"^[A-Za-z][A-Za-z0-9 _-]{0,40}:\s+", "", t)
    return t.strip()


def _chunks_from_speaker_segments(speaker_segments: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Build topic chunks from speaker-labeled segments (best for Zoom mode).
    Accepts segments like:
      {start,end,text,speaker_id/speaker,speaker_name}
    """
    if not speaker_segments:
        return []

    chunks: List[Dict[str, Any]] = []
    for seg in speaker_segments:
        if not isinstance(seg, dict):
            continue
        raw = (seg.get("text") or "").strip()
        if len(raw) < 20:
            continue

        cleaned = _clean_chunk(_strip_speaker_prefix(raw))
        if len(cleaned) < 20:
            continue

        chunks.append(
            {
                "text": cleaned,
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
                "speaker_id": seg.get("speaker_id") or seg.get("speaker") or "UNKNOWN",
                "speaker_name": seg.get("speaker_name"),
            }
        )

    return chunks


def _chunks_from_sentences(text: str) -> List[Dict[str, Any]]:
    """
    Fallback: split transcript into sentence-like chunks.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Split into sentence-ish pieces
    parts = re.split(r"[.!?]+", text)
    parts = [_clean_chunk(_strip_speaker_prefix(p)) for p in parts]
    parts = [p for p in parts if len(p) >= 20]

    return [{"text": p} for p in parts]


def segment_topics(text: str, speaker_segments=None, num_topics: int = 3) -> Dict[str, Any]:
    """
    Unsupervised topic segmentation using TF-IDF + KMeans.

    Zoom-safe behavior:
      - Prefer speaker-labeled segments if available (better chunks).
      - Strip "Name:" prefixes before vectorization.
      - Robust fallbacks for short transcripts / few chunks.
      - Return JSON-safe Python primitives (no numpy types).
    """
    print("Segmenting topics...")

    text = (text or "").strip()
    if not text:
        print("✗ Topic segmentation skipped (empty transcript)")
        return {"topics": [], "num_topics": 0}

    num_topics = max(1, int(num_topics or 1))

    # If transcript is tiny, avoid clustering
    if len(text.split()) < 40:
        print("⚠ Transcript too short for reliable topic segmentation – using a single topic")
        summary = text[:500]
        return {"topics": [{"topic_id": 0, "summary": summary, "sentence_count": 1}], "num_topics": 1}

    # Prefer speaker segments if present (works great in Zoom mode)
    chunks = _chunks_from_speaker_segments(speaker_segments)

    # Fallback to sentence splitting
    if not chunks:
        chunks = _chunks_from_sentences(text)

    if len(chunks) < 3:
        print("⚠ Not enough chunks for topic segmentation – using a single topic")
        joined = " ".join([c["text"] for c in chunks]) if chunks else text
        return {
            "topics": [{"topic_id": 0, "summary": joined[:500], "sentence_count": max(1, len(chunks))}],
            "num_topics": 1,
        }

    # If too few chunks to cluster meaningfully, single topic
    if len(chunks) <= num_topics:
        print("⚠ Too few distinct chunks to form multiple topics – using a single topic")
        joined = " ".join([c["text"] for c in chunks])
        return {"topics": [{"topic_id": 0, "summary": joined[:500], "sentence_count": len(chunks)}], "num_topics": 1}

    try:
        texts = [c["text"] for c in chunks]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=300,
            min_df=1,
            ngram_range=(1, 2),
        )
        X = vectorizer.fit_transform(texts)

        # Conservative cluster count: never more than half the chunks
        n_clusters = min(num_topics, max(1, len(chunks) // 2))
        if n_clusters <= 1:
            print("⚠ Data not rich enough for clustering – using a single topic")
            joined = " ".join(texts)
            return {"topics": [{"topic_id": 0, "summary": joined[:500], "sentence_count": len(chunks)}], "num_topics": 1}

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        model.fit(X)

        topics: List[Dict[str, Any]] = []
        labels = model.labels_

        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            if cluster_indices.size == 0:
                continue

            centroid = model.cluster_centers_[i]
            cluster_vecs = X[cluster_indices]

            sims = cluster_vecs.dot(centroid)
            sims = sims.A1 if hasattr(sims, "A1") else np.array(sims).reshape(-1)

            ranked_local = np.argsort(-sims)
            ranked_indices = cluster_indices[ranked_local]

            top_idxs = ranked_indices[:3].tolist()
            top_texts = [texts[j] for j in top_idxs]

            topic_text = ". ".join(top_texts)
            if len(topic_text) > 500:
                topic_text = topic_text[:500].rstrip() + "..."

            topics.append(
                {
                    "topic_id": int(i),
                    "summary": topic_text,
                    "sentence_count": int(cluster_indices.size),
                }
            )

        topics.sort(key=lambda t: t["sentence_count"], reverse=True)

        print(f"✓ Segmented into {len(topics)} topics")
        return {"topics": topics, "num_topics": int(len(topics))}

    except Exception as e:
        print(f"✗ Topic segmentation error: {e}")
        return {"topics": [{"topic_id": 0, "summary": text[:500], "sentence_count": 1}], "num_topics": 1}
