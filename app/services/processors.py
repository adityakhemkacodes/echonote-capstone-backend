# app/services/processors.py

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import sys
import cv2

# Ensure project root is on path so "app.*" imports work
sys.path.append(str(Path(__file__).parent.parent))

from app.modules.identity_resolution import resolve_display_name
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

from app.modules.identity_fusion import (
    fuse_identities,
    fuse_identities_from_zoom,
    _name_similarity,
)


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
    # INTERNAL MODE DETECTION
    # -------------------------------------------------------------------------
    def _is_zoom_speaker_mode(self) -> bool:
        """
        True ONLY when we actually have Zoom per-speaker exports
        (speaker_name and/or source_file present), not diarization SPEAKER_00 style.
        """
        tx = self.results.get("transcription", {}) or {}
        labeled = tx.get("speaker_labeled_segments", []) or []
        if not labeled:
            return False

        for seg in labeled[:25]:
            if not isinstance(seg, dict):
                continue
            if seg.get("speaker_name") or seg.get("source_file"):
                return True

        return False

    def _build_zoom_speaker_registry(self) -> List[Dict[str, Any]]:
        tx = self.results.get("transcription", {}) or {}
        labeled = tx.get("speaker_labeled_segments", []) or []

        seen = set()
        registry: List[Dict[str, Any]] = []
        for seg in labeled:
            if not isinstance(seg, dict):
                continue
            sid = str(seg.get("speaker_id") or "").strip()
            if not sid or sid.upper() == "UNKNOWN":
                continue
            if sid in seen:
                continue
            seen.add(sid)

            name = str(seg.get("speaker_name") or "").strip()
            registry.append({"speaker_id": sid, "name": name or None})

        return registry

    def _build_person_id_to_name(self) -> Dict[str, str]:
        participants = self.results.get("participants", {}) or {}
        person_to_name = participants.get("person_to_name") or {}
        alias_to_person = participants.get("alias_to_person") or {}

        if not isinstance(person_to_name, dict):
            person_to_name = {}
        if not isinstance(alias_to_person, dict):
            alias_to_person = {}

        out: Dict[str, str] = {}

        for k, v in person_to_name.items():
            if v and str(k).startswith("PERSON_"):
                out[str(k)] = str(v).strip()

        for face_pid, person_pid in alias_to_person.items():
            nm = person_to_name.get(person_pid)
            if nm and str(nm).strip():
                out[str(face_pid)] = str(nm).strip()

        return out

    def _count_diarization_speakers(self) -> int:
        """
        Legacy mode: pyannote diarization speakers (SPEAKER_00/01/...) are the best
        available 'participant count' when Zoom per-speaker files are absent.
        """
        tx = self.results.get("transcription", {}) or {}
        segs = tx.get("speaker_segments", []) or []
        ids = set()
        for s in segs:
            if not isinstance(s, dict):
                continue
            sid = (s.get("speaker") or s.get("speaker_id") or "").strip()
            if sid:
                ids.add(sid)
        return len(ids)

    # -------------------------------------------------------------------------
    # STEP 1: TRANSCRIPTION
    # -------------------------------------------------------------------------
    def process_transcription(self) -> Optional[Dict[str, Any]]:
        print("\n=== Step 1: Transcription ===")
        try:
            result = transcribe_with_speakers(self.video_path)

            transcript: str = ""
            speaker_segments: List[Dict[str, Any]] = []
            whisper_segments: List[Dict[str, Any]] = []

            if not isinstance(result, tuple):
                raise TypeError("transcribe_with_speakers did not return a tuple.")

            if len(result) == 3:
                transcript, speaker_segments, whisper_segments = result
            elif len(result) == 2:
                transcript, speaker_segments = result
                whisper_segments = []
            else:
                raise ValueError(
                    f"transcribe_with_speakers returned {len(result)} values, expected 2 or 3."
                )

            speaker_labeled_segments: List[Dict[str, Any]] = []

            # Zoom mode: already speaker-labeled utterances
            if (
                speaker_segments
                and isinstance(speaker_segments[0], dict)
                and ("speaker_id" in speaker_segments[0])
                and ("text" in speaker_segments[0])
            ):
                for s in speaker_segments:
                    text = (s.get("text") or "").strip()
                    if not text:
                        continue
                    speaker_labeled_segments.append(
                        {
                            "start": float(s.get("start", 0.0) or 0.0),
                            "end": float(s.get("end", 0.0) or 0.0),
                            "text": text,
                            "speaker_id": s.get("speaker_id") or "UNKNOWN",
                            "speaker_name": s.get("speaker_name"),
                            "source_file": s.get("source_file"),
                        }
                    )

                whisper_segments = [
                    {"start": x["start"], "end": x["end"], "text": x["text"]}
                    for x in speaker_labeled_segments
                ]
            else:
                speaker_labeled_segments = self._align_speakers_to_transcript(
                    whisper_segments, speaker_segments
                )

            self.results["transcription"] = {
                "full_text": (transcript or "").strip(),
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
    # STEP 2: PARTICIPANTS
    # -------------------------------------------------------------------------
    def process_participants(self) -> Optional[Dict[str, Any]]:
        print("\n=== Step 2: Participant Identification ===")
        try:
            face_data = track_faces(self.video_path)
            rough_face_track_count = count_participants(face_data)

            transcription = self.results.get("transcription", {}) or {}
            transcript_text = transcription.get("full_text", "") or ""

            if self._is_zoom_speaker_mode():
                print("✓ Zoom per-speaker mode detected → Zoom-first fusion + best-effort face alignment.")

                speaker_registry = self._build_zoom_speaker_registry()
                speaker_labeled_segments = transcription.get("speaker_labeled_segments", []) or []

                participants_fused, speaker_map, person_map = fuse_identities_from_zoom(
                    face_tracks=face_data,
                    speaker_registry=speaker_registry,
                    speaker_labeled_segments=speaker_labeled_segments,
                    auto_align_faces=True,
                )

                participants_fused = participants_fused or []
                speaker_map = speaker_map or {}
                person_map = person_map or {}

                for p in participants_fused:
                    if not isinstance(p, dict):
                        continue
                    pid = str(p.get("participant_id") or "").strip()
                    nm = (p.get("canonical_name") or "").strip()
                    if pid.startswith("PERSON_") and nm and pid not in person_map:
                        person_map[pid] = nm

                alias_to_person: Dict[str, str] = {}
                for p in participants_fused:
                    if not isinstance(p, dict):
                        continue
                    pid = str(p.get("participant_id") or "").strip()
                    face_pid = str(p.get("face_person_id") or "").strip()
                    if pid.startswith("PERSON_") and face_pid:
                        alias_to_person[face_pid] = pid

                entities = extract_entities(transcript_text)

                participant_count = max(len(participants_fused), 0)

                detected_names = [
                    p.get("canonical_name")
                    for p in participants_fused
                    if isinstance(p, dict)
                    and isinstance(p.get("canonical_name"), str)
                    and p.get("canonical_name").strip()
                ]

                self.results["participants"] = {
                    "count": participant_count,
                    "participants": participants_fused,
                    "participants_fused": participants_fused,
                    "speaker_to_person": speaker_map,
                    "person_to_name": person_map,
                    "alias_to_person": alias_to_person,
                    "detected_names": detected_names[:participant_count] if participant_count else detected_names,
                    "ocr_names": [],
                    "face_tracking_data": face_data,
                    "face_track_count": int(rough_face_track_count or 0),
                    "entities": entities[:50],
                    "mode": "zoom_per_speaker",
                    "speaker_registry": speaker_registry,
                }
                return self.results["participants"]

            # -------------------------
            # Legacy OCR + diarization
            # -------------------------
            print("ℹ Legacy mode (no per-speaker audio detected) → using OCR + identity fusion.")

            transcript_names = detect_participant_names(transcript_text)

            video_names = extract_names_for_tracks(self.video_path, face_data)
            if not video_names:
                video_names = extract_names_from_video(self.video_path)

            ocr_names = video_names or []
            ocr_name_list = [
                n.get("name") for n in ocr_names
                if isinstance(n, dict) and n.get("name")
            ]

            speaker_segments = transcription.get("speaker_segments", []) or []

            participants_fused, speaker_map, person_map = fuse_identities(
                face_data,
                ocr_names,
                speaker_segments,
                transcript_names=transcript_names,
            )

            entities = extract_entities(transcript_text)

            # --- augment missing names using transcript
            existing_canon = [p.get("canonical_name") for p in participants_fused if p.get("canonical_name")]
            existing_canon_lower = {c.lower() for c in existing_canon if isinstance(c, str)}

            transcript_names_unique: List[str] = []
            for n in (transcript_names or []):
                if n and n not in transcript_names_unique:
                    transcript_names_unique.append(n)

            candidate_text_names: List[str] = []
            for name in transcript_names_unique:
                ln = name.strip().lower()
                is_duplicate = False
                for ec in existing_canon_lower:
                    if _name_similarity(ln, ec) >= 0.85:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    candidate_text_names.append(name)

            anon_participants = [p for p in participants_fused if not p.get("canonical_name")]
            for p, name in zip(anon_participants, candidate_text_names):
                p["canonical_name"] = name
                aliases = p.get("aliases") or []
                if name not in aliases:
                    aliases.append(name)
                p["aliases"] = aliases

            for p in participants_fused:
                cname = p.get("canonical_name")
                if cname:
                    person_map[p["participant_id"]] = cname

            canonical_names = [p["canonical_name"] for p in participants_fused if p.get("canonical_name")]
            raw_names = list(set((transcript_names or []) + ocr_name_list))

            base_names: List[str] = []
            base_names.extend(canonical_names)

            for n in raw_names:
                if not n:
                    continue
                if any(n.strip().lower() == c.strip().lower() for c in canonical_names):
                    continue
                base_names.append(n)

            cleaned_detected: List[str] = []
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

                is_redundant = False
                for existing in cleaned_detected:
                    le = existing.strip().lower()
                    if lower_n == le:
                        is_redundant = True
                        break
                    if _name_similarity(lower_n, le) >= 0.85:
                        is_redundant = True
                        break

                if not is_redundant:
                    cleaned_detected.append(n)
                    seen_lower.add(lower_n)

            diar_speaker_count = self._count_diarization_speakers()
            participant_count = max(diar_speaker_count, len(cleaned_detected), 1)

            max_names = participant_count if participant_count else len(cleaned_detected)
            detected_names_for_display = cleaned_detected[:max_names]

            self.results["participants"] = {
                "count": participant_count,
                "participants": participants_fused,
                "participants_fused": participants_fused,
                "speaker_to_person": speaker_map,
                "person_to_name": person_map,
                "alias_to_person": {},
                "detected_names": detected_names_for_display,
                "ocr_names": ocr_name_list[:10],
                "face_tracking_data": face_data,
                "face_track_count": int(rough_face_track_count or 0),
                "diarization_speaker_count": diar_speaker_count,
                "entities": entities[:50],
                "mode": "legacy_ocr_fusion",
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
        print("\n=== Step 3: Sentiment Analysis ===")
        try:
            face_data = self.results.get("participants", {}).get("face_tracking_data", []) or []
            person_id_to_name = self._build_person_id_to_name()

            emotions_timeline, overall_facial_raw = analyze_facial_sentiment(
                self.video_path,
                face_data,
                person_id_to_name=person_id_to_name,
            )

            transcript_text = self.results.get("transcription", {}).get("full_text", "") or ""
            speaker_labeled_segments = self.results.get("transcription", {}).get("speaker_labeled_segments", []) or []

            text_sentiment_raw = analyze_text_sentiment(
                transcript_text,
                speaker_labeled_segments,
            )

            # --- Calibrate/pad both streams (stronger facial padding) ---
            overall_facial_cal = self._calibrate_facial_distribution(overall_facial_raw)
            text_sentiment_cal = self._calibrate_text_sentiment(text_sentiment_raw)

            # Facial-only "overall mood" for display (as requested)
            overall_mood = self._build_facial_only_overall_mood(overall_facial_cal, text_sentiment_cal)

            self.results["sentiment"] = {
                "facial_sentiment": {
                    "emotions_timeline": emotions_timeline,
                    "overall_raw": overall_facial_raw,
                    "overall_calibrated": overall_facial_cal,
                },
                "text_sentiment": {
                    "raw": text_sentiment_raw,
                    "calibrated": text_sentiment_cal,
                },
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
        print("\n=== Step 4: Timeline & Mood Changes ===")
        try:
            sentiment_data = self.results.get("sentiment", {}) or {}
            transcript_data = self.results.get("transcription", {}) or {}
            participants_data = self.results.get("participants", {}) or {}

            speaker_labeled_segments = transcript_data.get("speaker_labeled_segments", []) or []
            speaker_to_person = participants_data.get("speaker_to_person", {}) or {}
            person_to_name = participants_data.get("person_to_name", {}) or {}
            alias_to_person = participants_data.get("alias_to_person", {}) or {}

            mood_changes = detect_mood_changes(
                sentiment_data.get("facial_sentiment", {}) or {},
                sentiment_data.get("text_sentiment", {}) or {},
                transcript_data.get("speaker_segments", []) or [],
                speaker_labeled_segments,
                speaker_to_person,
                person_to_name,
                alias_to_person=alias_to_person,
            )

            timeline = create_timeline(
                transcript_data.get("speaker_segments", []) or [],
                mood_changes,
                participants_data.get("face_tracking_data", []) or [],
            )

            for item in timeline:
                if not isinstance(item, dict):
                    continue

                if item.get("type") == "speech":
                    sid = item.get("speaker")
                    if not item.get("speaker_name"):
                        display = resolve_display_name(
                            speaker_id=sid,
                            person_id=None,
                            speaker_to_person=speaker_to_person,
                            person_to_name=person_to_name,
                        )
                        if display:
                            item["speaker_name"] = display
                            item["display_name"] = display

                elif item.get("type") == "mood_change":
                    if not item.get("display_name"):
                        sid = item.get("speaker_id")
                        pid = item.get("person_id")
                        display = resolve_display_name(
                            speaker_id=sid,
                            person_id=pid,
                            speaker_to_person=speaker_to_person,
                            person_to_name=person_to_name,
                        )
                        if display:
                            item["speaker_name"] = display
                            item["display_name"] = display

            zoom_mode = self._is_zoom_speaker_mode()
            timebase = "global_video"
            timestamp_semantics = (
                "Zoom per-speaker mode: speech segments may be aligned to meeting time, but exports can include padding/rebasing; "
                "facial/mood timestamps are global video seconds."
                if zoom_mode
                else "All timestamps are global video seconds."
            )

            self.results["timeline"] = {
                "mood_changes": mood_changes,
                "full_timeline": timeline[:100],
                "key_moments": self._extract_key_moments(mood_changes),
                "timebase": timebase,
                "timestamp_semantics": timestamp_semantics,
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
        print("\n=== Step 5: Topic Segmentation ===")
        try:
            transcript_text = self.results.get("transcription", {}).get("full_text", "") or ""
            speaker_segments = self.results.get("transcription", {}).get("speaker_segments", []) or []
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

        try:
            summary = generate_summary(
                transcript,
                self.results.get("sentiment", {}) or {},
                self.results.get("topics", {}) or {},
                self.results.get("participants", {}) or {},
            )
        except Exception as e:
            print("✗ Summary error:", e)
            summary = {"main_summary": "(summary unavailable)", "key_topics": []}

        try:
            action_items = extract_action_items(
                transcript,
                self.results.get("transcription", {}).get("speaker_labeled_segments", []) or [],
                self.results.get("participants", {}) or {},
            )
        except Exception as e:
            print("✗ Action item error:", e)
            action_items = []

        self.results["insights"] = {
            "summary": summary,
            "action_items": action_items,
        }
        self.results["insights"]["meeting_metrics"] = self._calculate_metrics()
        return self.results["insights"]

    # -------------------------------------------------------------------------
    # ORCHESTRATOR
    # -------------------------------------------------------------------------
    def process_all(self, save_output: bool = True) -> Dict[str, Any]:
        print(f"\n{'=' * 60}")
        print("Starting Meeting Analysis")
        print(f"Video: {self.video_path}")
        print(f"{'=' * 60}")

        self.process_transcription()
        self.process_participants()
        self.process_sentiment()
        self.process_timeline()
        self.process_topics()
        self.generate_insights()

        self.results["status"] = "completed"
        self.results["completed_at"] = datetime.now().isoformat()

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

    def _clamp(self, x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        try:
            x = float(x)
        except Exception:
            return lo
        return max(lo, min(hi, x))

    def _normalize_pct_dist(self, dist: Dict[str, float]) -> Dict[str, float]:
        # dist is in percent space (0..100)
        clean: Dict[str, float] = {}
        total = 0.0
        for k, v in (dist or {}).items():
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            if fv < 0:
                fv = 0.0
            clean[str(k)] = fv
            total += fv
        if total <= 0:
            return clean
        return {k: round((v / total) * 100.0, 2) for k, v in clean.items()}

    def _calibrate_facial_distribution(self, overall_facial_raw: Dict[str, Any]) -> Dict[str, float]:
        """
        Stronger padding:
        - reduce "sad" aggressively (false negatives are common)
        - push most of that mass into neutral, some into happy
        """
        raw = overall_facial_raw or {}
        # expect keys: neutral, sad, happy (but handle extras safely)
        neutral = float(raw.get("neutral", 0.0) or 0.0)
        sad = float(raw.get("sad", 0.0) or 0.0)
        happy = float(raw.get("happy", 0.0) or 0.0)

        # --- stronger pad knobs ---
        # shrink sadness to 30% of its value, but never below a tiny floor
        sad_scaled = max(0.5, sad * 0.30)

        removed = max(0.0, sad - sad_scaled)

        # allocate removed mass: mostly neutral, some happy
        neutral_boost = removed * 0.7
        happy_boost = removed * 0.3

        neutral2 = neutral + neutral_boost
        happy2 = happy + happy_boost
        sad2 = sad_scaled

        # keep any other emotions (if model adds them) but downweight them a bit
        extras: Dict[str, float] = {}
        for k, v in raw.items():
            if k in ("neutral", "sad", "happy"):
                continue
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            # treat other negatives (angry/fear/disgust) similarly to sad → shrink
            extras[str(k)] = max(0.0, fv * 0.50)

        out = {"neutral": neutral2, "happy": happy2, "sad": sad2, **extras}
        return self._normalize_pct_dist(out)

    def _calibrate_text_sentiment(self, text_sentiment_raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Text padding:
        - distilbert SST-2 tends to mark sarcasm/jokes as negative
        - We bias toward neutral/positive, and require stronger evidence for negative.
        Works off text_sentiment_raw["overall"]["distribution"] if present.
        """
        ts = text_sentiment_raw or {}
        overall = ts.get("overall", {}) if isinstance(ts, dict) else {}
        dist = overall.get("distribution", {}) if isinstance(overall, dict) else {}

        pos = float(dist.get("positive", 0.0) or 0.0)
        neg = float(dist.get("negative", 0.0) or 0.0)
        neu = float(dist.get("neutral", 0.0) or 0.0)

        # If model doesn't provide neutral (common for SST-2), infer
        if neu <= 0.0 and (pos + neg) > 0.0:
            # interpret "uncertainty" as neutral-ish
            neu = max(0.0, 1.0 - (pos + neg))
        if (pos + neg + neu) <= 0:
            pos, neg, neu = 0.34, 0.33, 0.33

        # --- padding knobs ---
        # shrink negative significantly unless it is overwhelming
        # (we don't have utterance-level logits here, so keep it simple)
        neg_scaled = neg * 0.55

        removed = max(0.0, neg - neg_scaled)

        # allocate removed: mostly neutral, some positive
        neu2 = neu + removed * 0.70
        pos2 = pos + removed * 0.30
        neg2 = neg_scaled

        total = pos2 + neg2 + neu2
        if total > 0:
            pos2 /= total
            neg2 /= total
            neu2 /= total

        # rebuild overall
        dominant = "neutral"
        mx = max(pos2, neg2, neu2)
        if mx == pos2:
            dominant = "positive"
        elif mx == neg2:
            dominant = "negative"
        else:
            dominant = "neutral"

        cal = json.loads(json.dumps(ts)) if isinstance(ts, dict) else {}
        cal.setdefault("overall", {})
        cal["overall"]["dominant"] = dominant
        cal["overall"]["distribution"] = {
            "positive": round(pos2, 3),
            "negative": round(neg2, 3),
            "neutral": round(neu2, 3),
        }
        return cal

    def _build_facial_only_overall_mood(self, facial_calibrated: Dict[str, float], text_calibrated: Dict[str, Any]) -> Dict[str, Any]:
        # facial_calibrated is percent space
        f = facial_calibrated or {}
        # pick dominant from facial
        dom = "neutral"
        best = -1.0
        for k, v in f.items():
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            if fv > best:
                best = fv
                dom = str(k)

        # text distribution (0..1)
        text_overall = (text_calibrated or {}).get("overall", {}) if isinstance(text_calibrated, dict) else {}
        text_dist = text_overall.get("distribution", {}) if isinstance(text_overall, dict) else {}

        return {
            "dominant_emotion": dom,
            "confidence": round(float(best / 100.0), 3) if best >= 0 else 0.0,
            "facial_distribution": f,  # calibrated facial shown in terminal
            "text_sentiment": text_overall,  # calibrated text overall, shown separately in terminal
            "text_distribution": {
                "positive": float(text_dist.get("positive", 0.0) or 0.0),
                "negative": float(text_dist.get("negative", 0.0) or 0.0),
                "neutral": float(text_dist.get("neutral", 0.0) or 0.0),
            },
        }

    def _extract_key_moments(self, mood_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not mood_changes:
            return []
        sorted_changes = sorted(mood_changes, key=lambda x: x.get("impact_score", 0), reverse=True)
        return sorted_changes[:5]

    def _align_speakers_to_transcript(
        self,
        whisper_segments: List[Dict[str, Any]],
        speaker_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not whisper_segments:
            return []

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

    def _calculate_metrics(self, action_items_override: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        speaker_segments = self.results.get("transcription", {}).get("speaker_segments", []) or []

        duration = 0.0
        if speaker_segments:
            try:
                duration = max(
                    [float(s.get("end", 0) or 0) for s in speaker_segments if isinstance(s, dict)]
                )
            except Exception:
                duration = 0.0

        if not duration:
            try:
                cap = cv2.VideoCapture(self.video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                cap.release()

                if fps > 0 and frame_count > 0:
                    duration = frame_count / fps
                    print(f"ℹ Duration inferred from video metadata: {duration:.2f} seconds")
                else:
                    print("⚠ Could not infer duration from video (fps/frame_count missing)")
            except Exception as e:
                print(f"⚠ Error while inferring duration from video: {e}")

        if action_items_override is not None:
            action_items_list = action_items_override
        else:
            insights = self.results.get("insights")
            action_items_list = insights.get("action_items", []) if isinstance(insights, dict) else []

        topics_obj = self.results.get("topics", {}) or {}
        topics_count = len(topics_obj.get("topics", []) or []) if isinstance(topics_obj, dict) else 0

        return {
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2) if duration else 0.0,
            "participant_count": self.results.get("participants", {}).get("count", 0),
            "total_words": self.results.get("transcription", {}).get("word_count", 0),
            "mood_changes": len(self.results.get("timeline", {}).get("mood_changes", []) or []),
            "action_items_count": len(action_items_list or []),
            "topics_discussed": topics_count,
        }

    def _save_results(self) -> str:
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
        return self.results


def process_meeting_video(video_path: str, save_output: bool = True) -> Dict[str, Any]:
    processor = MeetingProcessor(video_path)
    return processor.process_all(save_output=save_output)
