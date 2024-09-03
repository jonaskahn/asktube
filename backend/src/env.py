import os
import tempfile

# API KEYS
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", None)

# WHISPER MODELS
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

APP_DIR = os.getenv("APP_DIR", os.path.join(tempfile.gettempdir(), "asktube"))
