# app/main.py
from fastapi import FastAPI, UploadFile
from app.services import (
    transcribe, summarizer, sentiment, entity_recognition, topic_segmentation, timeline
)

app = FastAPI()

@app.post("/analyze_meeting")
async def analyze_meeting(file: UploadFile):
    audio_path = f"app/static/{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    transcript, speakers = transcribe.transcribe_with_speakers(audio_path)
    summary = summarizer.generate_summary(transcript)
    sentiments = sentiment.analyze_text_sentiment(transcript)
    entities = entity_recognition.extract_entities(transcript)
    topics = topic_segmentation.segment_topics(transcript)
    full_timeline, key_moments = timeline.build_meeting_timeline(speakers, sentiments, [])

    return {
        "summary": summary,
        "topics": topics,
        "entities": entities,
        "sentiments": sentiments,
        "timeline": full_timeline,
        "key_moments": key_moments
    }
