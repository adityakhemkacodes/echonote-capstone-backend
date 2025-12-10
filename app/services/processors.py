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
from app.modules.name_detection import (
    detect_participant_names,
    extract_names_from_video,
    extract_names_for_tracks,
)
from app.modules.face_tracking import track_faces, count_participants
from app.modules.facial_sentiment import analyze_facial_sentiment
from app.modules.sentiment import analyze_text_sentiment
from app.modules.topic_segmentation import segment_topics
from app.modules.timeline import create_timeline, detect_mood_changes
from app.modules.summarizer import generate_summary, extract_action_items
from app.modules.identity_fusion import fuse_identities, _name_similarity


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
            rough_participant_count = count_participants(face_data)

            # 2) Transcript-based name detection (NER / heuristics)
            transcript_text = self.results.get("transcription", {}).get("full_text", "")
            transcript_names = detect_participant_names(transcript_text)

            # 3) OCR names from video overlay, using per-face label strips
            #    Structure: [{"timestamp": float, "name": "Some Name"}, ...]
            video_names = extract_names_for_tracks(self.video_path, face_data)

            # Fallback: if per-face OCR fails for some reason, use the older global method
            if not video_names:
                video_names = extract_names_from_video(self.video_path)

            ocr_names = video_names
            ocr_name_list = [n["name"] for n in ocr_names]

            # 4) Speaker segments from diarization (time ranges only)
            transcription = self.results.get("transcription", {}) or {}
            speaker_segments = transcription.get("speaker_segments", []) or []

            # 5) Fuse faces + OCR names + speakers (+ transcript names) into unified identities
            participants_fused, speaker_map, person_map = fuse_identities(
                face_data,
                ocr_names,
                speaker_segments,
                transcript_names=transcript_names,
            )

            # 6) Entity recognition for additional context
            entities = extract_entities(transcript_text)

            # 6.1) Try to assign transcript-only names to anonymous participants
            existing_canon = [
                p["canonical_name"]
                for p in participants_fused
                if p.get("canonical_name")
            ]
            existing_canon_lower = {c.lower() for c in existing_canon if c}

            # Deduplicate transcript_names while preserving order
            transcript_names_unique: list[str] = []
            for n in (transcript_names or []):
                if not n:
                    continue
                if n not in transcript_names_unique:
                    transcript_names_unique.append(n)

            # Filter transcript names that are not just variants of existing canonicals
            candidate_text_names: list[str] = []
            for name in transcript_names_unique:
                ln = name.strip().lower()
                is_duplicate = False
                for ec in existing_canon_lower:
                    # If it's very similar to an existing canonical, skip (e.g. "Aditya" vs "Aditya Khemka")
                    if _name_similarity(ln, ec) >= 0.85:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    candidate_text_names.append(name)

            # Assign candidate transcript names to participants that don't yet have a canonical_name
            anon_participants = [p for p in participants_fused if not p.get("canonical_name")]

            for p, name in zip(anon_participants, candidate_text_names):
                p["canonical_name"] = name
                aliases = p.get("aliases") or []
                if name not in aliases:
                    aliases.append(name)
                p["aliases"] = aliases

            # Refresh person_to_name map with any newly assigned canonical names
            for p in participants_fused:
                cname = p.get("canonical_name")
                if cname:
                    person_map[p["participant_id"]] = cname

            # 7) Build a global "detected names" list for easy display:
            #    - Prefer canonical per-person names
            #    - Also include transcript-based names that aren't just aliases/typos
            canonical_names = [
                p["canonical_name"]
                for p in participants_fused
                if p.get("canonical_name")
            ]

            # Raw pool from transcript + OCR
            raw_names = list(set((transcript_names or []) + ocr_name_list))

            # Start with canonical names, then extend with raw names that aren't just duplicates
            base_names: list[str] = []
            base_names.extend(canonical_names)

            for n in raw_names:
                if not n:
                    continue
                if any(n.strip().lower() == c.strip().lower() for c in canonical_names):
                    continue
                base_names.append(n)

            # Deduplicate and avoid alias-y forms like "Aditya", "Khemka", "Khemks"
            cleaned_detected: list[str] = []
            seen_lower = set()

            for n in base_names:
                if not n:
                    continue
                n = n.strip()
                if not n:
                    continue

                lower_n = n.lower()
                if lower_n in seen_lower:
                    continue

                tokens_n = lower_n.split()
                is_redundant = False

                for existing in cleaned_detected:
                    le = existing.strip().lower()
                    tokens_e = le.split()

                    # Exact match
                    if lower_n == le:
                        is_redundant = True
                        break

                    # If this is a single-token alias that matches FIRST or LAST name
                    # of an already-known full name, skip it
                    # (e.g. "aditya" vs "aditya khemka", or "khemka" vs "aditya khemka").
                    if len(tokens_n) == 1 and tokens_e:
                        if tokens_n[0] == tokens_e[0] or tokens_n[0] == tokens_e[-1]:
                            is_redundant = True
                            break

                    # If very similar (typo variants like "khemks"), treat as redundant too.
                    if _name_similarity(lower_n, le) >= 0.85:
                        is_redundant = True
                        break

                if not is_redundant:
                    cleaned_detected.append(n)
                    seen_lower.add(lower_n)

            # Prefer fused participant count (faces+fusion); fall back to rough face-based estimate
            participant_count = len(participants_fused) if participants_fused else rough_participant_count

            # Cap displayed names to participant count
            max_names = participant_count if participant_count else len(cleaned_detected)
            detected_names_for_display = cleaned_detected[:max_names]

            # 8) Final participants block
            self.results["participants"] = {
                "count": participant_count,
                "participants": participants_fused,         # unified identities with canonical_name/aliases
                "participants_fused": participants_fused,   # backward-compatible alias
                "speaker_to_person": speaker_map,           # "SPEAKER_00" -> "PERSON_1"
                "person_to_name": person_map,               # "PERSON_1" -> "Aditya Khemka" (canonical)
                "detected_names": detected_names_for_display,
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
            speaker_labeled_segments = self.results.get("transcription", {}).get(
                "speaker_labeled_segments", []
            )

            text_sentiment = analyze_text_sentiment(
                transcript_text,
                speaker_labeled_segments,
            )

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
            participants_data = self.results.get("participants", {})

            # Extra info for context-aware mood changes
            speaker_labeled_segments = transcript_data.get("speaker_labeled_segments", [])
            speaker_to_person = participants_data.get("speaker_to_person", {})
            person_to_name = participants_data.get("person_to_name", {})

            mood_changes = detect_mood_changes(
                sentiment_data.get("facial_sentiment", {}),
                sentiment_data.get("text_sentiment", {}),
                transcript_data.get("speaker_segments", []),
                speaker_labeled_segments,
                speaker_to_person,
                person_to_name,
            )

            timeline = create_timeline(
                transcript_data.get("speaker_segments", []),
                mood_changes,
                participants_data.get("face_tracking_data", []),
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
    # STEP 6: SUMMARY & ACTIONABLE INSIGHTS (Gemini)
    # -------------------------------------------------------------------------
    def generate_insights(self) -> Dict[str, Any]:
        print("\n=== Step 6: Generating Insights (Gemini) ===")

        transcript = self.results.get("transcription", {}).get("full_text", "") or ""

        # --- SUMMARY ---
        try:
            summary = generate_summary(
                transcript,
                self.results.get("sentiment", {}),
                self.results.get("topics", {}),
                self.results.get("participants", {}),
            )
        except Exception as e:
            print("✗ Summary error:", e)
            summary = {"main_summary": "(summary unavailable)", "key_topics": []}

        # --- ACTION ITEMS ---
        try:
            action_items = extract_action_items(
                transcript,
                self.results.get("transcription", {}).get("speaker_labeled_segments", []),
                self.results.get("participants", {}),
            )
        except Exception as e:
            print("✗ Action item error:", e)
            action_items = []

        self.results["insights"] = {
            "summary": summary,
            "action_items": action_items,
        }

        # metrics AFTER summary & actions
        self.results["insights"]["meeting_metrics"] = self._calculate_metrics()

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
        """
        Calculate combined mood from facial and text sentiment.

        - Facial: neutral / happy / sad / angry / fear / disgust (percentages)
        - Text: positive / negative / neutral (0–1)
        - Composite: weighted mix so one noisy channel doesn't dominate.
        """
        facial_overall = facial_sentiment.get("overall", {}) or {}
        text_overall = text_sentiment.get("overall", {}) or {}
        text_dist = text_overall.get("distribution", {}) or {}

        # Text sentiment (0–1)
        t_pos = float(text_dist.get("positive", 0.0))
        t_neg = float(text_dist.get("negative", 0.0))
        t_neu = float(text_dist.get("neutral", 0.0))

        # Decide a "text tone" label, but treat near-ties as neutral/balanced
        text_max = max(t_pos, t_neg, t_neu) if (t_pos or t_neg or t_neu) else 0.0
        if text_max < 0.55 or abs(t_pos - t_neg) < 0.20:
            text_tone_label = "NEUTRAL"
        elif t_pos > t_neg:
            text_tone_label = "POSITIVE"
        else:
            text_tone_label = "NEGATIVE"

        # Facial: treat as percentages (0–100) → convert to 0–1
        f_happy = float(facial_overall.get("happy", 0.0)) / 100.0
        f_neutral = float(facial_overall.get("neutral", 0.0)) / 100.0
        f_neg = (
            float(facial_overall.get("sad", 0.0))
            + float(facial_overall.get("angry", 0.0))
            + float(facial_overall.get("fear", 0.0))
            + float(facial_overall.get("disgust", 0.0))
        ) / 100.0

        # Composite distribution:
        #  - text is usually more reliable for "tone" → 60%
        #  - facial is complementary → 40%
        w_text = 0.6
        w_face = 0.4

        comp_pos = w_text * t_pos + w_face * f_happy
        comp_neg = w_text * t_neg + w_face * f_neg
        comp_neu = w_text * t_neu + w_face * f_neutral

        total = comp_pos + comp_neg + comp_neu
        if total > 0:
            comp_pos /= total
            comp_neg /= total
            comp_neu /= total

        # Final dominant label from composite
        if comp_pos >= comp_neg and comp_pos >= comp_neu:
            overall_label = "positive"
            confidence = comp_pos
        elif comp_neg >= comp_pos and comp_neg >= comp_neu:
            overall_label = "negative"
            confidence = comp_neg
        else:
            overall_label = "neutral"
            confidence = comp_neu

        return {
            "dominant_emotion": overall_label,  # "positive" | "negative" | "neutral"
            "confidence": round(float(confidence), 3),
            "facial_distribution": facial_overall,
            "text_sentiment": text_overall,
            "text_distribution": {
                "positive": round(t_pos, 3),
                "negative": round(t_neg, 3),
                "neutral": round(t_neu, 3),
            },
            "composite_distribution": {
                "positive": round(float(comp_pos), 3),
                "negative": round(float(comp_neg), 3),
                "neutral": round(float(comp_neu), 3),
            },
            "raw": {
                "text_tone_label": text_tone_label,
            },
        }


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

    def _calculate_metrics(self, action_items_override: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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

        # Work out action items – prefer the ones we just computed if provided
        if action_items_override is not None:
            action_items_list = action_items_override
        else:
            action_items_list = []
            insights = self.results.get("insights")
            if isinstance(insights, dict):
                action_items_list = insights.get("action_items", []) or []

        metrics = {
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2) if duration else 0.0,
            "participant_count": self.results.get("participants", {}).get("count", 0),
            "total_words": self.results.get("transcription", {}).get("word_count", 0),
            "mood_changes": len(
                self.results.get("timeline", {}).get("mood_changes", [])
            )
            if self.results.get("timeline")
            else 0,
            "action_items_count": len(action_items_list),
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
