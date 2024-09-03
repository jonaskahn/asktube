import os

# API KEYS
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", None)
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", None)
CLAUDE_API_KEY=os.getenv("CLAUDE_API_KEY", None)

# WHISPER MODELS
WHISPER_MODEL=os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE=os.getenv("WHISPER_DEVICE", "cpu")

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/storage")
SQL_DATABASE = os.getenv("SQL_DATABASE", os.path.join(STORAGE_PATH, "database.sqlite3"))
VECTOR_DATABASE = os.getenv("VECTOR_DATABASE", os.path.join(STORAGE_PATH, "vectors"))
