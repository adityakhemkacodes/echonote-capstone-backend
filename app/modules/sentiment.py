# app/modules/sentiment.py

from typing import List, Dict, Any, Optional
from transformers import pipeline

_sentiment_pipeline = None


def _get_sentiment_pipeline():
    """
    Lazy-load a small, fast sentiment model.
    DistilBERT SST-2 is binary (pos/neg) but we'll turn it
    into a soft pos/neg/neutral distribution.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("Loading text sentiment model (distilbert-base-uncased-finetuned-sst-2-english)...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    return _sentiment_pipeline


def analyze_text_sentiment(
    transcript: str,
    speaker_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Analyze text sentiment.

    - If speaker_segments are provided (from speaker-labeled Whisper segments),
      we score each utterance separately and aggregate.
    - Otherwise we run sentiment on the whole transcript as a single chunk.

    Returns:
      {
        "overall": {
          "label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
          "score": float,  # confidence for the dominant label
          "distribution": {
             "positive": float,
             "negative": float,
             "neutral": float
          }
        },
        "by_segment": [
          {
            "start": float,
            "end": float,
            "speaker_id": str,
            "text": str,
            "sentiment": {
               "label": ...,
               "scores": {
                 "positive": float,
                 "negative": float,
                 "neutral": float
               }
            },
          },
          ...
        ]
      }
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return {
            "overall": {
                "label": "NEUTRAL",
                "score": 0.0,
                "distribution": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            },
            "by_segment": [],
        }

    pipe = _get_sentiment_pipeline()

    segments = speaker_segments or []

    # Build the texts we will score
    batched_texts = []
    meta: List[Dict[str, Any]] = []

    if segments:
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text or len(text) < 4:
                continue
            batched_texts.append(text)
            meta.append(
                {
                    "start": float(seg.get("start", 0.0) or 0.0),
                    "end": float(seg.get("end", 0.0) or 0.0),
                    "speaker_id": seg.get("speaker_id", "UNKNOWN"),
                    "text": text,
                }
            )
    else:
        text = transcript
        batched_texts.append(text)
        meta.append(
            {
                "start": 0.0,
                "end": 0.0,
                "speaker_id": "UNKNOWN",
                "text": text,
            }
        )

    if not batched_texts:
        return {
            "overall": {
                "label": "NEUTRAL",
                "score": 0.0,
                "distribution": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            },
            "by_segment": [],
        }

    raw_results = pipe(
        batched_texts,
        truncation=True,
        max_length=256,
        batch_size=16,
    )

    overall_pos = overall_neg = overall_neu = 0.0
    total_weight = 0.0
    by_segment: List[Dict[str, Any]] = []

    for seg_meta, res in zip(meta, raw_results):
        label = (res.get("label") or "").upper()
        score = float(res.get("score", 0.5))

        # Binary model → convert to soft pos/neg
        if label == "POSITIVE":
            pos = score
            neg = 1.0 - score
        else:
            neg = score
            pos = 1.0 - score

        # Turn uncertainty into "neutral" mass
        # - Large score → confident → less neutral
        # - ~0.5 score → very neutral
        max_p = max(pos, neg)
        neutral = 1.0 - max_p

        # Reduce extremes a bit so we don't report 0.99 negative for a chill meeting
        pos = max(0.0, pos - neutral * 0.25)
        neg = max(0.0, neg - neutral * 0.25)

        # Normalize so pos+neg+neutral <= 1 (we keep neutral as "uncertainty")
        # No strict renorm – just leave as is for interpretability.

        # Weight by utterance length
        weight = max(1, len(seg_meta["text"].split()))
        overall_pos += pos * weight
        overall_neg += neg * weight
        overall_neu += neutral * weight
        total_weight += weight

        # Decide per-segment label
        if pos >= neg and pos >= neutral:
            seg_label = "POSITIVE"
        elif neg >= pos and neg >= neutral:
            seg_label = "NEGATIVE"
        else:
            seg_label = "NEUTRAL"

        by_segment.append(
            {
                "start": seg_meta["start"],
                "end": seg_meta["end"],
                "speaker_id": seg_meta["speaker_id"],
                "text": seg_meta["text"],
                "sentiment": {
                    "label": seg_label,
                    "scores": {
                        "positive": round(pos, 3),
                        "negative": round(neg, 3),
                        "neutral": round(neutral, 3),
                    },
                },
            }
        )

    if total_weight <= 0:
        total_weight = 1.0

    overall_pos /= total_weight
    overall_neg /= total_weight
    overall_neu /= total_weight

    # Overall label by argmax
    if overall_pos >= overall_neg and overall_pos >= overall_neu:
        overall_label = "POSITIVE"
        overall_score = overall_pos
    elif overall_neg >= overall_pos and overall_neg >= overall_neu:
        overall_label = "NEGATIVE"
        overall_score = overall_neg
    else:
        overall_label = "NEUTRAL"
        overall_score = overall_neu

    return {
        "overall": {
            "label": overall_label,
            "score": round(float(overall_score), 3),
            "distribution": {
                "positive": round(float(overall_pos), 3),
                "negative": round(float(overall_neg), 3),
                "neutral": round(float(overall_neu), 3),
            },
        },
        "by_segment": by_segment,
    }
