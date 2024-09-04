import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE", "on")
APP_DIR = os.getenv("APP_DIR", os.path.join(tempfile.gettempdir(), "asktube"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", None)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
LANGUAGE_DETECT_SEGMENT_LENGTH: int = int(os.getenv("LANGUAGE_DETECT_SEGMENT_LENGTH", 30))
LANGUAGE_PREFER_USAGE: str = os.getenv("LANGUAGE_PREFER_USAGE", "en")
VIDEO_CHUNK_LENGTH: int = int(os.getenv("VIDEO_CHUNK_LENGTH", 600))
