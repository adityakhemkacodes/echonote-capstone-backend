# app/modules/sentiment.py
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Initialize model once
_sentiment_model = None

def get_sentiment_model():
    """Lazy load sentiment model"""
    global _sentiment_model
    if _sentiment_model is None:
        print("Loading sentiment analysis model...")
        _sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _sentiment_model


def analyze_text_sentiment(text: str, speaker_segments: list = None):
    """
    Analyze text sentiment using transformer model.
    Returns sentiment for the full text and per-segment if provided.
    """
    print("Analyzing text sentiment...")
    
    text = (text or "").strip()
    if not text:
        print("✗ Text sentiment skipped (empty transcript)")
        return {
            'overall': {'label': 'NEUTRAL', 'score': 0.0},
            'segments': []
        }

    try:
        model = get_sentiment_model()
        
        # Analyze overall text (truncate if too long)
        text_chunk = text[:512] if len(text) > 512 else text
        overall_result = model(text_chunk)[0]
        
        results = {
            'overall': {
                'label': overall_result['label'],
                'score': round(overall_result['score'], 3)
            },
            'segments': []
        }
        
        # Analyze per speaker segment if provided
        if speaker_segments:
            for seg in speaker_segments[:20]:  # Limit to first 20 segments
                seg_text = seg.get('text', '')
                if seg_text and len(seg_text) > 10:
                    chunk = seg_text[:512]
                    seg_result = model(chunk)[0]
                    results['segments'].append({
                        'speaker': seg.get('speaker', 'Unknown'),
                        'start': seg.get('start', 0),
                        'end': seg.get('end', 0),
                        'sentiment': seg_result['label'],
                        'confidence': round(seg_result['score'], 3)
                    })
        
        print(f"✓ Text sentiment analysis complete")
        return results
        
    except Exception as e:
        print(f"✗ Text sentiment error: {e}")
        return {
            'overall': {'label': 'NEUTRAL', 'score': 0.0},
            'segments': []
        }


def analyze_facial_sentiment(video_analysis_results):
    """
    Aggregate facial emotion stats from tracked faces.
    """
    if not video_analysis_results:
        return {
            'happy': 0,
            'neutral': 100,
            'sad': 0
        }
    
    emotion_counts = {}
    for frame in video_analysis_results:
        if "emotion" in frame:
            emotion = frame["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    total = sum(emotion_counts.values()) or 1
    return {k: round(v / total * 100, 2) for k, v in emotion_counts.items()}