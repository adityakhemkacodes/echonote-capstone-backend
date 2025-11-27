import cv2
import mediapipe as mp
from typing import List, Dict

mp_face = mp.solutions.face_detection


def _iou(box1, box2) -> float:
    """
    Intersection-over-Union between two [x, y, w, h] boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area_a = w1 * h1
    area_b = w2 * h2
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def track_faces(video_path: str, sample_rate: int = 5) -> List[Dict]:
    """
    Detect faces in video and return tracking data with STABLE person_ids.

    sample_rate: process every Nth frame for performance

    Return:
      [
        {
          "timestamp": float,
          "bbox": [x, y, w, h],
          "person_id": "PERSON_0",
          "confidence": float
        },
        ...
      ]
    """
    print("Tracking faces in video...")

    try:
        cap = cv2.VideoCapture(video_path)
        face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

        face_tracks: List[Dict] = []
        frame_id = 0

        # Internal track state
        active_tracks = []  # [{ "person_id": "PERSON_0", "last_bbox": [...], "last_ts": float }]
        next_person_idx = 0

        MAX_TIME_GAP = 2.0      # seconds – don't link detections too far apart
        MIN_IOU_FOR_MATCH = 0.4 # how strict we are when linking to an existing track

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_id % sample_rate == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector.process(rgb)

                if results.detections:
                    h, w, _ = frame.shape

                    for det in results.detections:
                        bboxC = det.location_data.relative_bounding_box
                        bbox = [
                            int(bboxC.xmin * w),
                            int(bboxC.ymin * h),
                            int(bboxC.width * w),
                            int(bboxC.height * h),
                        ]

                        # Ensure valid bbox
                        if bbox[2] <= 0 or bbox[3] <= 0:
                            continue

                        # Try to match to an existing track
                        best_track = None
                        best_iou = 0.0

                        for track in active_tracks:
                            # ignore tracks too far in time
                            if abs(timestamp - track["last_ts"]) > MAX_TIME_GAP:
                                continue
                            iou = _iou(bbox, track["last_bbox"])
                            if iou > best_iou:
                                best_iou = iou
                                best_track = track

                        if best_track is not None and best_iou >= MIN_IOU_FOR_MATCH:
                            person_id = best_track["person_id"]
                            best_track["last_bbox"] = bbox
                            best_track["last_ts"] = timestamp
                        else:
                            # create new track
                            person_id = f"PERSON_{next_person_idx}"
                            next_person_idx += 1
                            active_tracks.append(
                                {
                                    "person_id": person_id,
                                    "last_bbox": bbox,
                                    "last_ts": timestamp,
                                }
                            )

                        face_tracks.append(
                            {
                                "timestamp": float(timestamp),
                                "bbox": bbox,
                                "person_id": person_id,
                                "confidence": float(det.score[0]) if det.score else 0.0,
                            }
                        )

            frame_id += 1

            # Limit processing for performance (~30 seconds @ 30fps)
            if frame_id > 900:
                break

        cap.release()
        unique_ids = {t["person_id"] for t in face_tracks}
        print(
            f"✓ Face tracking complete: {len(face_tracks)} detections, "
            f"{len(unique_ids)} unique tracks"
        )
        return face_tracks

    except Exception as e:
        print(f"✗ Face tracking error: {e}")
        return []


def count_participants(face_tracks: List[Dict]) -> int:
    """
    Estimate number of unique participants from face tracking data.
    """
    if not face_tracks:
        return 0

    unique_persons = {track.get("person_id", "UNKNOWN") for track in face_tracks}
    count = len(unique_persons)
    print(f"✓ Estimated {count} participants")
    return count


def track_faces_and_names(video_path: str) -> List[Dict]:
    """
    (Optional helper) Detect faces and associate them with OCR-detected names.
    Not currently used by the main pipeline, but kept for debugging / future use.
    """
    from app.modules.name_detection import extract_names_from_video

    print("Tracking faces and matching names...")

    names = extract_names_from_video(video_path)
    face_tracks = track_faces(video_path)

    speaker_map = []
    for track in face_tracks:
        timestamp = track["timestamp"]
        nearest_name = None
        min_diff = float("inf")

        for n in names:
            time_diff = abs(n["timestamp"] - timestamp)
            if time_diff < 2 and time_diff < min_diff:
                nearest_name = n["name"]
                min_diff = time_diff

        speaker_map.append(
            {
                "timestamp": timestamp,
                "bbox": track["bbox"],
                "person_id": track["person_id"],
                "name": nearest_name or "Unknown",
            }
        )

    print(f"✓ Matched {len(speaker_map)} face-name associations")
    return speaker_map
