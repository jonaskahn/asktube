import os
import tempfile

DEBUG_MODE = os.getenv("DEBUG_MODE", "on")

APP_DIR = os.getenv("APP_DIR", os.path.join(tempfile.gettempdir(), "asktube"))

# API KEYS
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", None)

# WHISPER MODELS
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

# SETTINGS
LANGUAGE_DETECT_SEGMENT_LENGTH: int = os.getenv("LANGUAGE_DETECT_SEGMENT_LENGTH", 30)
LANGUAGE_PREFER_USAGE: str = os.getenv("LANGUAGE_PREFER_USAGE",
                                       "en")  # Get your own preference (set1): https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
VIDEO_CHUNK_LENGTH: int = os.getenv("VIDEO_CHUNK_LENGTH", 600)
