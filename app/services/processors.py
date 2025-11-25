# app/services/processors.py

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
import cv2

# Ensure project root is on path so "app.*" imports work
sys.path.append(str(Path(__file__).parent.parent))

# Import all processing modules
from app.modules.transcribe import transcribe_with_speakers
from app.modules.entity_recognition import extract_entities
from app.modules.name_detection import detect_participant_names, extract_names_from_video
from app.modules.face_tracking import track_faces, count_participants
from app.modules.facial_sentiment import analyze_facial_sentiment
from app.modules.sentiment import analyze_text_sentiment
from app.modules.topic_segmentation import segment_topics
from app.modules.timeline import create_timeline, detect_mood_changes
from app.modules.summarizer import generate_summary, extract_action_items
from app.modules.identity_fusion import fuse_identities


class MeetingProcessor:
    """Orchestrates the complete meeting analysis pipeline"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.results: Dict[str, Any] = {
            "video_path": video_path,
            "processed_at": datetime.now().isoformat(),
            "status": "initialized",
        }

    # -------------------------------------------------------------------------
    # STEP 1: TRANSCRIPTION
    # -------------------------------------------------------------------------
    def process_transcription(self) -> Optional[Dict[str, Any]]:
        """Step 1: Transcribe video and identify speakers"""
        print("\n=== Step 1: Transcription ===")
        try:
            result = transcribe_with_speakers(self.video_path)

            # Support both 2-tuple and 3-tuple return values
            transcript: str = ""
            speaker_segments: List[Dict[str, Any]] = []
            whisper_segments: List[Dict[str, Any]] = []

            if isinstance(result, tuple):
                if len(result) == 3:
                    transcript, speaker_segments, whisper_segments = result
                elif len(result) == 2:
                    transcript, speaker_segments = result
                else:
                    # Unexpected shape – treat as failure
                    raise ValueError(
                        f"transcribe_with_speakers returned {len(result)} values, expected 2 or 3."
                    )
            else:
                raise TypeError("transcribe_with_speakers did not return a tuple.")

            # Align Whisper time ranges with diarization segments to label speakers
            speaker_labeled_segments = self._align_speakers_to_transcript(
                whisper_segments, speaker_segments
            )

            self.results["transcription"] = {
                "full_text": transcript,
                "speaker_segments": speaker_segments,
                "word_count": len(transcript.split()) if transcript else 0,
                "whisper_segments": whisper_segments,
                "speaker_labeled_segments": speaker_labeled_segments,
            }
            return self.results["transcription"]

        except Exception as e:
            print(f"✗ Error in transcription: {str(e)}")
            self.results["transcription"] = {"error": str(e)}
            return None

    # -------------------------------------------------------------------------
    # STEP 2: PARTICIPANTS (FACE TRACKING + NAMES + IDENTITY FUSION)
    # -------------------------------------------------------------------------
    def process_participants(self) -> Optional[Dict[str, Any]]:
        """Step 2: Identify participants through face tracking and name detection"""
        print("\n=== Step 2: Participant Identification ===")
        try:
            # 1) Face tracking
            face_data = track_faces(self.video_path)
            participant_count = count_participants(face_data)

            # 2) Transcript-based name detection (NER / heuristics)
            transcript_text = self.results.get("transcription", {}).get("full_text", "")
            detected_names = detect_participant_names(transcript_text)

            # 3) OCR names from video overlay
            #    Structure: [{"timestamp": float, "name": "Some Name"}, ...]
            video_names = extract_names_from_video(self.video_path)
            ocr_names = video_names
            ocr_name_list = [n["name"] for n in ocr_names]

            # 4) Speaker segments from diarization (time ranges only)
            transcription = self.results.get("transcription", {}) or {}
            speaker_segments = transcription.get("speaker_segments", []) or []

            # 5) Fuse faces + OCR names + speakers into unified identities
            participants_fused, speaker_map, person_map = fuse_identities(
                face_data,
                ocr_names,
                speaker_segments,
            )

            # 6) Entity recognition for additional context
            entities = extract_entities(transcript_text)

            # 7) Build a clean combined name list (transcript names + OCR names)
            all_names = list({*(detected_names or []), *ocr_name_list})

            # 8) Save everything in a single participants block
            self.results["participants"] = {
                "count": participant_count,
                "participants_fused": participants_fused,   # unified identities
                "speaker_to_person": speaker_map,           # "SPEAKER_00" -> "PERSON_1"
                "person_to_name": person_map,               # "PERSON_1" -> "Aditya Khemka"
                "detected_names": all_names[:10],           # human-friendly list
                "ocr_names": ocr_name_list[:10],
                "face_tracking_data": face_data[:200],      # limit size
                "entities": entities[:50],                  # limit entities
            }

            return self.results["participants"]

        except Exception as e:
            print(f"✗ Error in participant processing: {str(e)}")
            self.results["participants"] = {"error": str(e), "count": 0}
            return None

    # -------------------------------------------------------------------------
    # STEP 3: SENTIMENT
    # -------------------------------------------------------------------------
    def process_sentiment(self) -> Optional[Dict[str, Any]]:
        """Step 3: Analyze sentiment from both facial expressions and text"""
        print("\n=== Step 3: Sentiment Analysis ===")
        try:
            # Get face tracking data
            face_data = self.results.get("participants", {}).get(
                "face_tracking_data", []
            )

            # Facial sentiment analysis (use more tracks; facial_sentiment handles sampling)
            emotions_timeline, overall_facial = analyze_facial_sentiment(
                self.video_path,
                face_data[:80],  # give it up to ~80 tracks; module limits internally
            )

            # Text sentiment analysis
            transcript_text = self.results.get("transcription", {}).get(
                "full_text", ""
            )
            speaker_segments = self.results.get("transcription", {}).get(
                "speaker_segments", []
            )

            text_sentiment = analyze_text_sentiment(transcript_text, speaker_segments)

            # Calculate overall meeting mood
            overall_mood = self._calculate_overall_mood(
                {"emotions_timeline": emotions_timeline, "overall": overall_facial},
                text_sentiment,
            )

            self.results["sentiment"] = {
                "facial_sentiment": {
                    "emotions_timeline": emotions_timeline,
                    "overall": overall_facial,
                },
                "text_sentiment": text_sentiment,
                "overall_mood": overall_mood,
            }
            return self.results["sentiment"]

        except Exception as e:
            print(f"✗ Error in sentiment processing: {str(e)}")
            self.results["sentiment"] = {"error": str(e)}
            return None

    # -------------------------------------------------------------------------
    # STEP 4: TIMELINE & MOOD CHANGES
    # -------------------------------------------------------------------------
    def process_timeline(self) -> Optional[Dict[str, Any]]:
        """Step 4: Create timeline and detect mood changes"""
        print("\n=== Step 4: Timeline & Mood Changes ===")
        try:
            sentiment_data = self.results.get("sentiment", {})
            transcript_data = self.results.get("transcription", {})

            mood_changes = detect_mood_changes(
                sentiment_data.get("facial_sentiment", {}),
                sentiment_data.get("text_sentiment", {}),
                transcript_data.get("speaker_segments", []),
            )

            timeline = create_timeline(
                transcript_data.get("speaker_segments", []),
                mood_changes,
                self.results.get("participants", {}).get("face_tracking_data", []),
            )

            self.results["timeline"] = {
                "mood_changes": mood_changes,
                "full_timeline": timeline[:100],  # Limit timeline size
                "key_moments": self._extract_key_moments(mood_changes),
            }
            return self.results["timeline"]

        except Exception as e:
            print(f"✗ Error in timeline processing: {str(e)}")
            self.results["timeline"] = {"error": str(e)}
            return None

    # -------------------------------------------------------------------------
    # STEP 5: TOPIC SEGMENTATION
    # -------------------------------------------------------------------------
    def process_topics(self) -> Optional[Dict[str, Any]]:
        """Step 5: Segment topics from the conversation"""
        print("\n=== Step 5: Topic Segmentation ===")
        try:
            transcript_text = self.results.get("transcription", {}).get(
                "full_text", ""
            )
            speaker_segments = self.results.get("transcription", {}).get(
                "speaker_segments", []
            )

            topics = segment_topics(transcript_text, speaker_segments)
            self.results["topics"] = topics
            return self.results["topics"]

        except Exception as e:
            print(f"✗ Error in topic processing: {str(e)}")
            self.results["topics"] = {"error": str(e)}
            return None

    # -------------------------------------------------------------------------
    # STEP 6: SUMMARY & ACTIONABLE INSIGHTS
    # -------------------------------------------------------------------------
    def generate_insights(self) -> Dict[str, Any]:
        """Step 6: Generate summary and actionable insights"""
        print("\n=== Step 6: Generating Insights ===")

        transcript_text = self.results.get("transcription", {}).get("full_text", "")

        summary = None
        action_items: List[Dict[str, Any]] = []
        error: Optional[str] = None

        # 1) Summary
        try:
            summary = generate_summary(
                transcript_text,
                self.results.get("sentiment", {}),
                self.results.get("topics", {}),
                self.results.get("participants", {}),
            )
        except Exception as e:
            print(f"✗ Error generating summary: {e}")
            error = error or str(e)

        # 2) Action items
        try:
            # Pass speaker segments (plain diarization); function is regex-based on text today
            action_items = extract_action_items(
                transcript_text,
                self.results.get("transcription", {}).get(
                    "speaker_labeled_segments", []
                ),
            )
        except Exception as e:
            print(f"✗ Error extracting action items: {e}")
            error = error or str(e)

        self.results["insights"] = {
            "summary": summary,
            "action_items": action_items,
            "meeting_metrics": self._calculate_metrics(),
        }
        if error:
            self.results["insights"]["error"] = error

        return self.results["insights"]

    # -------------------------------------------------------------------------
    # ORCHESTRATOR
    # -------------------------------------------------------------------------
    def process_all(self, save_output: bool = True) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        print(f"\n{'=' * 60}")
        print("Starting Meeting Analysis")
        print(f"Video: {self.video_path}")
        print(f"{'=' * 60}")

        # Run all processing steps
        self.process_transcription()
        self.process_participants()
        self.process_sentiment()
        self.process_timeline()
        self.process_topics()
        self.generate_insights()

        self.results["status"] = "completed"
        self.results["completed_at"] = datetime.now().isoformat()

        # Save results if requested
        if save_output:
            output_path = self._save_results()
            self.results["output_path"] = output_path

        print(f"\n{'=' * 60}")
        print("✓ Analysis Complete!")
        print(f"{'=' * 60}\n")

        return self.results

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------
    def _calculate_overall_mood(
        self, facial_sentiment: Dict[str, Any], text_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate combined mood from facial and text sentiment"""
        mood = {
            "dominant_emotion": "neutral",
            "confidence": 0.0,
            "facial_distribution": facial_sentiment.get("overall", {}),
            "text_sentiment": text_sentiment.get("overall", {}),
        }

        # Get dominant facial emotion
        facial_dist = facial_sentiment.get("overall", {})
        if facial_dist:
            dominant = max(facial_dist.items(), key=lambda x: x[1])
            mood["dominant_emotion"] = dominant[0]
            mood["confidence"] = dominant[1] / 100.0

        return mood

    def _extract_key_moments(self, mood_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract the most significant mood change moments"""
        if not mood_changes:
            return []

        sorted_changes = sorted(
            mood_changes, key=lambda x: x.get("impact_score", 0), reverse=True
        )
        return sorted_changes[:5]

    def _align_speakers_to_transcript(
        self,
        whisper_segments: List[Dict[str, Any]],
        speaker_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Align Whisper time-stamped segments with diarization segments to
        obtain a speaker-labeled transcript.

        Returns list of:
          {"start": float, "end": float, "text": str, "speaker_id": str}
        """
        if not whisper_segments:
            return []

        # No diarization: just return segments with UNKNOWN speaker
        if not speaker_segments:
            return [
                {
                    "start": float(ws.get("start", 0.0) or 0.0),
                    "end": float(ws.get("end", 0.0) or 0.0),
                    "text": (ws.get("text") or "").strip(),
                    "speaker_id": "UNKNOWN",
                }
                for ws in whisper_segments
            ]

        aligned: List[Dict[str, Any]] = []

        for ws in whisper_segments:
            ws_start = float(ws.get("start", 0.0) or 0.0)
            ws_end = float(ws.get("end", ws_start) or ws_start)
            text = (ws.get("text") or "").strip()
            if ws_end <= ws_start:
                continue

            best_speaker = "UNKNOWN"
            best_overlap = 0.0

            for sp in speaker_segments:
                s_start = float(sp.get("start", 0.0) or 0.0)
                s_end = float(sp.get("end", s_start) or s_start)
                if s_end <= s_start:
                    continue

                # Overlap between [ws_start, ws_end] and [s_start, s_end]
                overlap = max(0.0, min(ws_end, s_end) - max(ws_start, s_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = sp.get("speaker", "UNKNOWN")

            aligned.append(
                {
                    "start": ws_start,
                    "end": ws_end,
                    "text": text,
                    "speaker_id": best_speaker,
                }
            )

        return aligned

    def _infer_duration_seconds(self) -> float:
        """
        Infer meeting duration from available timelines when speaker segments are missing.
        """
        candidates: List[float] = []

        # From face tracking timestamps
        face_data = self.results.get("participants", {}).get("face_tracking_data", [])
        if face_data:
            try:
                max_face_ts = max(
                    f.get("timestamp", 0) for f in face_data if isinstance(f, dict)
                )
                candidates.append(max_face_ts)
            except Exception:
                pass

        # From facial emotion timeline
        emo_timeline = (
            self.results.get("sentiment", {})
            .get("facial_sentiment", {})
            .get("emotions_timeline", [])
        )
        if emo_timeline:
            try:
                max_emo_ts = max(
                    e.get("timestamp", 0) for e in emo_timeline if isinstance(e, dict)
                )
                candidates.append(max_emo_ts)
            except Exception:
                pass

        if not candidates:
            return 0.0

        return round(max(candidates), 2)

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate various meeting metrics"""
        speaker_segments = self.results.get("transcription", {}).get(
            "speaker_segments", []
        )

        # First try: duration from diarization / speaker segments
        duration = 0.0
        if speaker_segments:
            duration = max([s.get("end", 0) for s in speaker_segments])

        # Fallback: infer duration from video file itself
        if not duration:
            try:
                cap = cv2.VideoCapture(self.video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                cap.release()

                if fps > 0 and frame_count > 0:
                    duration = frame_count / fps
                    print(
                        f"ℹ Duration inferred from video metadata: {duration:.2f} seconds"
                    )
                else:
                    print("⚠ Could not infer duration from video (fps/frame_count missing)")
            except Exception as e:
                print(f"⚠ Error while inferring duration from video: {e}")

        metrics = {
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2) if duration else 0.0,
            "participant_count": self.results.get("participants", {}).get("count", 0),
            "total_words": self.results.get("transcription", {}).get("word_count", 0),
            "mood_changes": len(
                self.results.get("timeline", {}).get("mood_changes", [])
            ),
            "action_items_count": len(
                self.results.get("insights", {}).get("action_items", [])
            )
            if self.results.get("insights")
            else 0,
            "topics_discussed": len(
                self.results.get("topics", {}).get("topics", [])
            )
            if isinstance(self.results.get("topics", {}), dict)
            else 0,
        }
        return metrics

    def _save_results(self) -> str:
        """Save results to JSON file"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(self.video_path).stem
        output_file = output_dir / f"{video_name}_analysis_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_file}")
        return str(output_file)

    def get_results(self) -> Dict[str, Any]:
        """Get current processing results"""
        return self.results


def process_meeting_video(video_path: str, save_output: bool = True) -> Dict[str, Any]:
    """Convenience function to process a meeting video"""
    processor = MeetingProcessor(video_path)
    return processor.process_all(save_output=save_output)
