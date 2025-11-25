# app/modules/face_tracking.py
import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection


def track_faces(video_path: str, sample_rate: int = 5):
    """
    Detect faces in video and return tracking data.
    sample_rate: process every Nth frame for performance
    """
    print("Tracking faces in video...")
    
    try:
        cap = cv2.VideoCapture(video_path)
        face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        face_tracks = []
        frame_id = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Sample frames for performance
            if frame_id % sample_rate == 0:
                results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.detections:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    for idx, det in enumerate(results.detections):
                        bboxC = det.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        bbox = [
                            int(bboxC.xmin * w),
                            int(bboxC.ymin * h),
                            int(bboxC.width * w),
                            int(bboxC.height * h)
                        ]
                        
                        face_tracks.append({
                            "timestamp": timestamp,
                            "bbox": bbox,
                            "person_id": f"PERSON_{idx}",
                            "confidence": det.score[0]
                        })
            
            frame_id += 1
            
            # Limit processing for performance
            if frame_id > 900:  # Process first ~30 seconds at 30fps
                break
        
        cap.release()
        print(f"✓ Face tracking complete: {len(face_tracks)} detections")
        return face_tracks
        
    except Exception as e:
        print(f"✗ Face tracking error: {e}")
        return []


def count_participants(face_tracks):
    """
    Estimate number of unique participants from face tracking data.
    Simple approach: count unique person_ids
    """
    if not face_tracks:
        return 0
    
    unique_persons = set()
    for track in face_tracks:
        unique_persons.add(track.get('person_id', 'UNKNOWN'))
    
    count = len(unique_persons)
    print(f"✓ Estimated {count} participants")
    return count


def track_faces_and_names(video_path: str):
    """
    Detect faces and associate them with OCR-detected names.
    This is a more complex version that tries to match names to faces.
    """
    from app.modules.name_detection import extract_names_from_video
    
    print("Tracking faces and matching names...")
    
    names = extract_names_from_video(video_path)
    face_tracks = track_faces(video_path)
    
    # Simple matching: find nearest name in time (±2s window)
    speaker_map = []
    for track in face_tracks:
        timestamp = track['timestamp']
        nearest_name = None
        min_diff = float('inf')
        
        for n in names:
            time_diff = abs(n["timestamp"] - timestamp)
            if time_diff < 2 and time_diff < min_diff:
                nearest_name = n["name"]
                min_diff = time_diff
        
        speaker_map.append({
            "timestamp": timestamp,
            "bbox": track['bbox'],
            "person_id": track['person_id'],
            "name": nearest_name or "Unknown"
        })
    
    print(f"✓ Matched {len(speaker_map)} face-name associations")
    return speaker_map