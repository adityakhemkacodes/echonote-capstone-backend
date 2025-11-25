# app/modules/identity_fusion.py

import math
from collections import defaultdict, Counter

def fuse_identities(face_tracks, ocr_names, speaker_segments):
    """
    Fuse person_id (faces), OCR names, and speaker_id (diarization)
    into unified participant identities.

    face_tracks: [{timestamp, bbox, person_id}]
    ocr_names:   [{timestamp, name}]
    speaker_segments: [{start, end, speaker}]

    Returns:
      participants: list of dicts with:
        {
          "participant_id": "PERSON_1",
          "name": "Aditya Khemka",
          "speaker_id": "SPEAKER_00",
          "face_person_id": "PERSON_1"
        }

      speaker_to_person: mapping
      person_to_name: mapping
    """

    # ---------------------------
    # STEP 1 — Assign OCR names → person_id
    # ---------------------------
    person_name_votes = defaultdict(list)

    for name_entry in ocr_names:
        ts = name_entry["timestamp"]
        name = name_entry["name"]

        # Find face closest in time
        best_pid = None
        best_dt = 999

        for f in face_tracks:
            dt = abs(f["timestamp"] - ts)
            if dt < best_dt:
                best_dt = dt
                best_pid = f["person_id"]

        if best_pid:
            person_name_votes[best_pid].append(name)

    # Majority vote for each person
    person_to_name = {}
    for pid, names in person_name_votes.items():
        if names:
            # choose the most common name
            name = Counter(names).most_common(1)[0][0]
            person_to_name[pid] = name

    # ---------------------------
    # STEP 2 — Assign speaker_id → person_id
    # ---------------------------
    speaker_votes = defaultdict(list)

    for seg in speaker_segments:
        speaker_id = seg["speaker"]
        start = seg["start"]
        end = seg["end"]
        mid = (start + end) / 2

        # find closest visible face at mid timestamp
        best_pid = None
        best_dt = 999

        for f in face_tracks:
            dt = abs(f["timestamp"] - mid)
            if dt < best_dt:
                best_dt = dt
                best_pid = f["person_id"]

        if best_pid:
            speaker_votes[speaker_id].append(best_pid)

    speaker_to_person = {}
    for sid, pids in speaker_votes.items():
        if pids:
            # choose the most common person_id
            person_to_use = Counter(pids).most_common(1)[0][0]
            speaker_to_person[sid] = person_to_use

    # ---------------------------
    # STEP 3 — Build participant list
    # ---------------------------

    participants = []
    all_pids = set(f["person_id"] for f in face_tracks)

    for pid in all_pids:
        name = person_to_name.get(pid, None)

        # find connected speaker_id
        speaker_id = None
        for sid, spid in speaker_to_person.items():
            if spid == pid:
                speaker_id = sid

        participants.append({
            "participant_id": pid,
            "name": name,
            "face_person_id": pid,
            "speaker_id": speaker_id
        })

    return participants, speaker_to_person, person_to_name
