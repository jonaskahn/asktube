import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

DEBUG_MODE = os.getenv("ASKTUBE_DEBUG_MODE", "on")
APP_DIR = os.getenv("ASKTUBE_APP_DIR", os.path.join(tempfile.gettempdir(), "asktube"))
GEMINI_API_KEY = os.getenv("ASKTUBE_GEMINI_API_KEY", None)
OPENAI_API_KEY = os.getenv("ASKTUBE_OPENAI_API_KEY", None)
CLAUDE_API_KEY = os.getenv("ASKTUBE_CLAUDE_API_KEY", None)
WHISPER_MODEL = os.getenv("ASKTUBE_WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("ASKTUBE_WHISPER_DEVICE", "cpu")
LANGUAGE_DETECT_SEGMENT_LENGTH: int = int(os.getenv("ASKTUBE_LANGUAGE_DETECT_SEGMENT_LENGTH", 30))
LANGUAGE_PREFER_USAGE: str = os.getenv("ASKTUBE_LANGUAGE_PREFER_USAGE", "en")
VIDEO_CHUNK_LENGTH: int = int(os.getenv("ASKTUBE_VIDEO_CHUNK_LENGTH", 600))
