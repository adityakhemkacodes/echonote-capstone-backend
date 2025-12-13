# app/services/file_handler.py

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

UPLOAD_FOLDER = Path("uploads")

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "flv", "zip"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def _safe_extract_zip(z: zipfile.ZipFile, dest_dir: Path) -> None:
    """
    Safe zip extraction to prevent Zip Slip (path traversal).
    Only extracts members that resolve under dest_dir.
    """
    dest_dir = dest_dir.resolve()
    for member in z.infolist():
        # Skip weird entries
        name = member.filename
        if not name or name.endswith("/"):
            continue

        # Resolve where it would land
        target_path = (dest_dir / name).resolve()

        # If not inside dest_dir, refuse
        if dest_dir not in target_path.parents and target_path != dest_dir:
            raise ValueError(f"Unsafe zip member path: {name}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with z.open(member, "r") as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _pick_main_meeting_video(extracted_dir: Path) -> Optional[Path]:
    """
    Heuristic to choose the main meeting video from an extracted Zoom folder.

    Priority:
      1) zoom_0.mp4 if present anywhere (common Zoom naming)
      2) Largest .mp4
      3) Largest video of any allowed type
    """
    videos: List[Path] = []
    mp4s: List[Path] = []
    zoom0: List[Path] = []

    for p in extracted_dir.rglob("*"):
        if not p.is_file():
            continue
        if _is_video(p):
            videos.append(p)
            if p.suffix.lower() == ".mp4":
                mp4s.append(p)
                if p.name.lower() == "zoom_0.mp4":
                    zoom0.append(p)

    if zoom0:
        # pick largest zoom_0.mp4 if multiple
        zoom0.sort(key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)
        return zoom0[0]

    if mp4s:
        mp4s.sort(key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)
        return mp4s[0]

    if videos:
        videos.sort(key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)
        return videos[0]

    return None


class FileHandler:
    def __init__(self, upload_folder: str = None):
        self.upload_folder = Path(upload_folder) if upload_folder else UPLOAD_FOLDER
        self.upload_folder.mkdir(parents=True, exist_ok=True)

    def allowed_file(self, filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def validate_file(self, file: FileStorage) -> Tuple[bool, Optional[str]]:
        if not file:
            return False, "No file provided"
        if file.filename == "":
            return False, "No file selected"
        if not self.allowed_file(file.filename):
            return False, f"File type not allowed. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

        # size check (works for werkzeug FileStorage)
        try:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            if file_size > MAX_FILE_SIZE:
                return (
                    False,
                    f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.0f}MB",
                )
        except Exception:
            # If we can't determine size, don't hard fail
            pass

        return True, None

    def save_file(
        self, file: FileStorage
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Save uploaded file to upload folder.

        Returns:
          (success, filepath, error_message, meta)

        meta includes:
          {
            "upload_type": "video" | "zip",
            "extracted_dir": "...",   # only for zip
            "original_filename": "...",
            "zip_path": "...",        # only for zip
          }
        """
        is_valid, error_msg = self.validate_file(file)
        if not is_valid:
            return False, None, error_msg, None

        try:
            filename = secure_filename(file.filename)
            ext = Path(filename).suffix.lower()
            filepath = self.upload_folder / filename

            # Avoid collisions for the uploaded file itself
            counter = 1
            base_name = filepath.stem
            extension = filepath.suffix
            while filepath.exists():
                filepath = self.upload_folder / f"{base_name}_{counter}{extension}"
                counter += 1

            file.save(str(filepath))

            # ZIP upload: extract and return the main meeting video path
            if ext == ".zip":
                # IMPORTANT: file_id should be stable for this uploaded zip *file* path
                # Even if we had to rename to *_1.zip, use that stem (matches routes.py behavior).
                file_id = filepath.stem
                extracted_dir = (self.upload_folder / file_id)
                extracted_dir.mkdir(parents=True, exist_ok=True)

                try:
                    with zipfile.ZipFile(filepath, "r") as z:
                        _safe_extract_zip(z, extracted_dir)
                except Exception as e:
                    # clean partial extraction
                    try:
                        shutil.rmtree(extracted_dir, ignore_errors=True)
                    except Exception:
                        pass
                    return False, None, f"Error extracting zip: {str(e)}", None

                main_video = _pick_main_meeting_video(extracted_dir)
                if not main_video:
                    return (
                        False,
                        None,
                        "Zip extracted but no meeting video file was found inside.",
                        None,
                    )

                meta = {
                    "upload_type": "zip",
                    "extracted_dir": str(extracted_dir),
                    "original_filename": filename,
                    "zip_path": str(filepath),
                }

                # Optional: delete uploaded zip to save disk (keep if you want debugging)
                # Comment out if you prefer to keep original zip.
                try:
                    filepath.unlink(missing_ok=True)  # py3.8+ supports missing_ok? if not, wrapped below
                except TypeError:
                    try:
                        if filepath.exists():
                            filepath.unlink()
                    except Exception:
                        pass
                except Exception:
                    pass

                return True, str(main_video), None, meta

            # Normal single-video upload
            meta = {"upload_type": "video", "original_filename": filename}
            return True, str(filepath), None, meta

        except Exception as e:
            return False, None, f"Error saving file: {str(e)}", None

    def delete_file(self, filepath: str) -> Tuple[bool, Optional[str]]:
        """
        Delete a file or an extracted zip folder.
        - If filepath is inside uploads/<id>/..., delete uploads/<id>.
        - If filepath is directly uploads/<file>, delete that file.
        """
        try:
            path = Path(filepath)
            if not path.exists():
                return False, "File not found"

            if self.upload_folder in path.parents:
                # direct file under uploads/
                if path.parent == self.upload_folder and path.is_file():
                    path.unlink()
                    return True, None

                # inside uploads/<something>/...
                rel = path.relative_to(self.upload_folder)
                top = rel.parts[0] if rel.parts else None
                if top:
                    target_dir = self.upload_folder / top
                    if target_dir.exists() and target_dir.is_dir():
                        shutil.rmtree(target_dir, ignore_errors=True)

                    # Also delete any zip with same stem if present
                    zip_path = self.upload_folder / f"{top}.zip"
                    if zip_path.exists():
                        try:
                            zip_path.unlink()
                        except Exception:
                            pass
                    return True, None

            return False, "Invalid path"
        except Exception as e:
            return False, f"Error deleting file: {str(e)}"

    def cleanup_uploads(self) -> None:
        try:
            if self.upload_folder.exists():
                shutil.rmtree(self.upload_folder)
                self.upload_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error cleaning up uploads: {str(e)}")

    def get_file_info(self, filepath: str) -> Optional[dict]:
        path = Path(filepath)
        if not path.exists():
            return None

        return {
            "filename": path.name,
            "size": path.stat().st_size,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "extension": path.suffix,
            "path": str(path),
            "parent_dir": str(path.parent),
        }
