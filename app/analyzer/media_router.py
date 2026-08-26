from __future__ import annotations

import enum
from pathlib import Path
import os

from app.utils.logger import get_logger

logger = get_logger(__name__)

class MediaType(enum.Enum):
    """Enumerated type for media formats."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    UNKNOWN = "unknown"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".wmv", ".flv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500MB

class MediaRouter:
    """Detects media type and validates files before analysis."""
    
    @staticmethod
    def detect_type(file_path: str | Path) -> MediaType:
        """Detect media type from file extension.
        
        Args:
            file_path: Path to the media file.
            
        Returns:
            The detected MediaType.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext in IMAGE_EXTENSIONS:
            return MediaType.IMAGE
        elif ext in VIDEO_EXTENSIONS:
            return MediaType.VIDEO
        elif ext in AUDIO_EXTENSIONS:
            return MediaType.AUDIO
        else:
            return MediaType.UNKNOWN
            
    @staticmethod
    def validate_file(file_path: str | Path) -> Path:
        """Validate file exists and is readable, return resolved Path.
        
        Args:
            file_path: Path to the file to validate.
            
        Returns:
            Resolved Path object.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is unknown or file is too large.
        """
        path = Path(file_path).resolve()
        
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")
            
        if not path.is_file():
            logger.error(f"Not a regular file: {path}")
            raise ValueError(f"Path is not a regular file: {path}")
            
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            logger.error(f"File exceeds maximum size of 500MB: {path}")
            raise ValueError(f"File exceeds maximum size of 500MB: {path}")
            
        media_type = MediaRouter.detect_type(path)
        if media_type == MediaType.UNKNOWN:
            logger.error(f"Unknown or unsupported media type for file: {path}")
            raise ValueError(f"Unknown or unsupported media type for file: {path}")
            
        return path
