# EchoNote Backend (final build)

This is a local, open-source backend for EchoNote — processes a meeting video (MP4) and returns JSON with:
- detected participants (from on-screen name OCR and face tracking)
- facial sentiment timeline and overall percentages
- speaker-attributed transcript (Whisper + diarization)
- abstractive summary, entities, topics
- timeline and key moments, and action-item candidates

## Quick start (Windows)

### 1) Prerequisites
- Python 3.10+
- Git (optional)
- Tesseract OCR (install and ensure `tesseract` executable is in PATH)
  - https://github.com/tesseract-ocr/tesseract
- ffmpeg (useful for audio extraction)
  - https://ffmpeg.org/download.html
- (Optional) NVIDIA drivers + CUDA if you plan to use GPU acceleration

### 2) Create & activate a virtual environment
```powershell
cd path\to\echonote_final_backend
python -m venv venv
.env\Scripts\activate
```

### 3) Install packages (CPU-only)
```powershell
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4) Install packages (GPU - recommended if you have NVIDIA CUDA)
If you have CUDA installed, install PyTorch with the appropriate CUDA wheels first (see https://pytorch.org/get-started/locally/), for example:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 5) Run the server
```powershell
uvicorn app.main:app --reload
```
Open: http://127.0.0.1:8000/docs and use `POST /analyze_meeting` to upload a meeting MP4 file.

### Files & folders
- `app/` - the FastAPI application and modules
- `uploads/` - saved uploaded videos (temporary)
- `results/` - JSON results written by the pipeline
- `requirements.txt` - pip dependencies
- `sample_output.json` - example of a typical output

## Notes & tips
- First run will download model weights for Whisper, transformers, spaCy, etc.
- If `pyannote.audio` diarization causes trouble, fall back to a Whisper-only transcription mode (fast but less speaker-attributed).
- For the deadline, test on short videos (~1-3 minutes) to validate the pipeline faster.
