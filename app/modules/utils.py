# app/services/utils.py
import os
import re

def clean_text(text: str):
    """
    Basic cleanup for raw transcripts.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ensure_dir(path: str):
    """
    Create directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)

def get_file_extension(filename: str):
    return os.path.splitext(filename)[1].lower()
