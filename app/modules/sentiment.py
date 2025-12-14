# app/modules/sentiment.py

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple
from transformers import pipeline

_sentiment_pipeline = None

# Simple joke/sarcasm softener markers (cheap but effective)
_JOKE_RE = re.compile(
    r"\b(lol|lmao|rofl|haha|hahaha|hehe|jk|just kidding|kidding|funny|joke|priorities)\b|😂|🤣|😅",
    re.IGNORECASE,
)

# Also soften negativity when common “meeting talk” appears
_MEETING_TONE_RE = re.compile(
    r"\b(blocker|blocked|issue|bug|fix|urgent|problem|error|outdated|broken|can't|cannot)\b",
    re.IGNORECASE,
)


def _get_sentiment_pipeline():
    """
    Lazy-load a small, fast sentiment model.
    DistilBERT SST-2 is binary (pos/neg); we convert to a 3-way distribution.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("Loading text sentiment model (distilbert-base-uncased-finetuned-sst-2-english)...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    return _sentiment_pipeline


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    s = 0.0
    for v in dist.values():
        try:
            s += max(0.0, float(v))
        except Exception:
            s += 0.0
    if s <= 0:
        return {k: 0.0 for k in dist}
    return {k: max(0.0, float(v)) / s for k, v in dist.items()}


def _dominant(dist: Dict[str, float]) -> str:
    if not dist:
        return "neutral"
    return max(dist.items(), key=lambda kv: kv[1])[0]


def _binary_to_3way(label: str, score: float) -> Dict[str, float]:
    """
    Convert SST-2 output to {positive, negative, neutral} prob-ish.
    """
    label = (label or "").upper()
    score = float(score or 0.5)

    if label == "POSITIVE":
        pos = score
        neg = 1.0 - score
    else:
        neg = score
        pos = 1.0 - score

    # uncertainty -> neutral mass
    max_p = max(pos, neg)
    neu = max(0.0, 1.0 - max_p)

    # mild “de-extreming”
    pos = max(0.0, pos - neu * 0.25)
    neg = max(0.0, neg - neu * 0.25)

    return _normalize({"positive": pos, "negative": neg, "neutral": neu})


def _calibrate_meeting_text(dist: Dict[str, float], full_text: str) -> Dict[str, float]:
    """
    Stronger meeting-friendly calibration.
    This is intentionally more padded to avoid “everything is negative” syndrome.
    """
    d = _normalize(
        {
            "positive": float(dist.get("positive", 0.0) or 0.0),
            "negative": float(dist.get("negative", 0.0) or 0.0),
            "neutral": float(dist.get("neutral", 0.0) or 0.0),
        }
    )

    # 1) Strong prior: meetings are usually neutral unless clearly emotional
    # (pad negative down, push neutral up)
    d["negative"] *= 0.50
    d["neutral"] *= 1.35
    d["positive"] *= 1.10
    d = _normalize(d)

    # 2) If this looks like normal “issue/bug” talk, soften negative a bit further
    if _MEETING_TONE_RE.search(full_text or ""):
        d["negative"] *= 0.80
        d["neutral"] *= 1.10
        d = _normalize(d)

    # 3) Joke softener (laughter markers, “priorities”, etc.)
    if _JOKE_RE.search(full_text or ""):
        d["negative"] *= 0.65
        d["neutral"] *= 1.10
        d["positive"] *= 1.05
        d = _normalize(d)

    # 4) Guardrail: don’t allow “negative” to dominate unless it’s really decisive
    # If negative is high but neutral is meaningful, pull some mass back to neutral.
    if d["negative"] > 0.45 and d["neutral"] > 0.22:
        shift = min(0.12, d["negative"] - 0.45)
        d["negative"] -= shift
        d["neutral"] += shift
        d = _normalize(d)

    return d


def analyze_text_sentiment(
    transcript: str,
    speaker_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Returns BOTH:
      - raw: SST-2 converted to 3-way
      - calibrated: meeting-friendly padded distribution

    Backward-compatible keys:
      - overall (== calibrated overall)
      - by_segment (== calibrated per segment)
    """
    transcript = (transcript or "").strip()
    if not transcript:
        empty_overall = {
            "dominant": "neutral",
            "distribution": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
        }
        return {
            "raw": {"overall": empty_overall, "by_segment": []},
            "calibrated": {"overall": empty_overall, "by_segment": []},
            "overall": {
                "label": "NEUTRAL",
                "score": 1.0,
                "distribution": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            },
            "by_segment": [],
        }

    pipe = _get_sentiment_pipeline()
    segments = speaker_segments or []

    batched_texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    if segments:
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text or len(text) < 4:
                continue
            batched_texts.append(text)

            sid = seg.get("speaker_id")
            if sid is None:
                sid = seg.get("speaker", "UNKNOWN")

            meta.append(
                {
                    "start": float(seg.get("start", 0.0) or 0.0),
                    "end": float(seg.get("end", 0.0) or 0.0),
                    "speaker_id": sid or "UNKNOWN",
                    "text": text,
                }
            )
    else:
        batched_texts.append(transcript)
        meta.append({"start": 0.0, "end": 0.0, "speaker_id": "UNKNOWN", "text": transcript})

    if not batched_texts:
        empty_overall = {
            "dominant": "neutral",
            "distribution": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
        }
        return {
            "raw": {"overall": empty_overall, "by_segment": []},
            "calibrated": {"overall": empty_overall, "by_segment": []},
            "overall": {
                "label": "NEUTRAL",
                "score": 1.0,
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

    # Weighted aggregates
    raw_sum = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    cal_sum = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    total_weight = 0.0

    raw_by_segment: List[Dict[str, Any]] = []
    cal_by_segment: List[Dict[str, Any]] = []

    for seg_meta, res in zip(meta, raw_results):
        label = (res.get("label") or "").upper()
        score = float(res.get("score", 0.5))

        raw_dist = _binary_to_3way(label, score)
        cal_dist = _calibrate_meeting_text(raw_dist, seg_meta.get("text", ""))

        weight = max(1, len((seg_meta.get("text") or "").split()))
        total_weight += weight

        for k in raw_sum:
            raw_sum[k] += raw_dist.get(k, 0.0) * weight
            cal_sum[k] += cal_dist.get(k, 0.0) * weight

        raw_by_segment.append(
            {
                "start": seg_meta["start"],
                "end": seg_meta["end"],
                "speaker_id": seg_meta["speaker_id"],
                "text": seg_meta["text"],
                "overall": {
                    "dominant": _dominant(raw_dist),
                    "distribution": {
                        "positive": round(float(raw_dist["positive"]), 3),
                        "negative": round(float(raw_dist["negative"]), 3),
                        "neutral": round(float(raw_dist["neutral"]), 3),
                    },
                },
            }
        )

        cal_by_segment.append(
            {
                "start": seg_meta["start"],
                "end": seg_meta["end"],
                "speaker_id": seg_meta["speaker_id"],
                "text": seg_meta["text"],
                "sentiment": {
                    "label": _dominant(cal_dist).upper(),
                    "scores": {
                        "positive": round(float(cal_dist["positive"]), 3),
                        "negative": round(float(cal_dist["negative"]), 3),
                        "neutral": round(float(cal_dist["neutral"]), 3),
                    },
                },
            }
        )

    if total_weight <= 0:
        total_weight = 1.0

    raw_overall = _normalize({k: raw_sum[k] / total_weight for k in raw_sum})
    cal_overall = _normalize({k: cal_sum[k] / total_weight for k in cal_sum})

    raw_overall_obj = {
        "dominant": _dominant(raw_overall),
        "distribution": {
            "positive": round(float(raw_overall["positive"]), 3),
            "negative": round(float(raw_overall["negative"]), 3),
            "neutral": round(float(raw_overall["neutral"]), 3),
        },
    }

    cal_overall_obj = {
        "dominant": _dominant(cal_overall),
        "distribution": {
            "positive": round(float(cal_overall["positive"]), 3),
            "negative": round(float(cal_overall["negative"]), 3),
            "neutral": round(float(cal_overall["neutral"]), 3),
        },
    }

    # Backward-compatible "overall" / "by_segment" (use calibrated)
    dom = cal_overall_obj["dominant"].upper()
    score = float(cal_overall[cal_overall_obj["dominant"]])

    return {
        "raw": {"overall": raw_overall_obj, "by_segment": raw_by_segment},
        "calibrated": {"overall": cal_overall_obj, "by_segment": cal_by_segment},

        # legacy-compatible keys:
        "overall": {
            "label": dom,
            "score": round(score, 3),
            "distribution": cal_overall_obj["distribution"],
        },
        "by_segment": cal_by_segment,
    }
