import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

UPLOAD_FOLDER = Path("uploads")
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

class FileHandler:
    def __init__(self, upload_folder: str = None):
        self.upload_folder = Path(upload_folder) if upload_folder else UPLOAD_FOLDER
        self.upload_folder.mkdir(parents=True, exist_ok=True)
    
    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def validate_file(self, file: FileStorage) -> Tuple[bool, Optional[str]]:
        """Validate uploaded file"""
        if not file:
            return False, "No file provided"
        
        if file.filename == '':
            return False, "No file selected"
        
        if not self.allowed_file(file.filename):
            return False, f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        
        # Check file size if possible
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return False, f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024)}MB"
        
        return True, None
    
    def save_file(self, file: FileStorage) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Save uploaded file to upload folder
        Returns: (success, filepath, error_message)
        """
        is_valid, error_msg = self.validate_file(file)
        if not is_valid:
            return False, None, error_msg
        
        try:
            filename = secure_filename(file.filename)
            filepath = self.upload_folder / filename
            
            # Handle duplicate filenames
            counter = 1
            base_name = filepath.stem
            extension = filepath.suffix
            while filepath.exists():
                filepath = self.upload_folder / f"{base_name}_{counter}{extension}"
                counter += 1
            
            file.save(str(filepath))
            return True, str(filepath), None
        
        except Exception as e:
            return False, None, f"Error saving file: {str(e)}"
    
    def delete_file(self, filepath: str) -> Tuple[bool, Optional[str]]:
        """Delete a file from the upload folder"""
        try:
            path = Path(filepath)
            if path.exists() and path.parent == self.upload_folder:
                path.unlink()
                return True, None
            return False, "File not found or invalid path"
        except Exception as e:
            return False, f"Error deleting file: {str(e)}"
    
    def cleanup_uploads(self) -> None:
        """Remove all files from upload folder"""
        try:
            if self.upload_folder.exists():
                shutil.rmtree(self.upload_folder)
                self.upload_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error cleaning up uploads: {str(e)}")
    
    def get_file_info(self, filepath: str) -> dict:
        """Get information about a file"""
        path = Path(filepath)
        if not path.exists():
            return None
        
        return {
            'filename': path.name,
            'size': path.stat().st_size,
            'size_mb': round(path.stat().st_size / (1024*1024), 2),
            'extension': path.suffix,
            'path': str(path)
        }