import re
from pathlib import Path
from typing import List, Dict

def extract_names_from_audio_files(audio_files: List[str]) -> List[Dict[str, str]]:
    """
    Extract participant names from Zoom individual audio filenames.
    Returns: [{"file": "path", "name": "John Doe"}, ...]
    """
    names = []
    patterns = [
        r"audio[_-]([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # audio_John_Doe
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)_audio",    # John_Doe_audio
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\.",       # John_Doe.m4a
    ]
    
    for audio_file in audio_files:
        filename = Path(audio_file).stem
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                name = match.group(1).replace("_", " ").strip()
                names.append({"file": audio_file, "name": name})
                break
    
    return names
