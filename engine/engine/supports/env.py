import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEBUG_MODE = os.getenv("AT_DEBUG_MODE", "on")
APP_DIR = os.path.abspath(os.getenv("AT_APP_DIR", os.path.join(tempfile.gettempdir(), "asktube")))
LANGUAGE_PREFER_USAGE: str = os.getenv("AT_LANGUAGE_PREFER_USAGE", "en")
AUDIO_CHUNK_RECOGNIZE_DURATION: int = int(os.getenv("AT_AUDIO_CHUNK_RECOGNIZE_DURATION", 30))
AUDIO_CHUNK_RECOGNIZE_THRESHOLD: int = int(os.getenv("AT_AUDIO_CHUNK_RECOGNIZE_THRESHOLD", 120))
AUDIO_CHUNK_CHAPTER_DURATION: int = int(os.getenv("AT_AUDIO_CHUNK_CHAPTER_DURATION", 600))
QUERY_SIMILAR_THRESHOLD: float = float(os.getenv("AT_QUERY_SIMILAR_THRESHOLD", 0.4))
TOKEN_CONTEXT_THRESHOLD: int = int(os.getenv("AT_TOKEN_CONTEXT_THRESHOLD", 8192))
AUDIO_ENHANCE_ENABLED: str = os.getenv("AT_AUDIO_ENHANCE_ENABLED", "off")
RAG_QUERY_IMPLEMENTATION: str = os.getenv("AT_RAG_QUERY_IMPLEMENTATION", "multiquery")
RAG_AUTO_SWITCH: str = os.getenv("AT_RAG_AUTO_SWITCH", "on")

GEMINI_API_KEY = os.getenv("AT_GEMINI_API_KEY", None)
OPENAI_API_KEY = os.getenv("AT_OPENAI_API_KEY", None)
CLAUDE_API_KEY = os.getenv("AT_CLAUDE_API_KEY", None)
VOYAGEAI_API_KEY = os.getenv("AT_VOYAGEAI_API_KEY", None)
MISTRAL_API_KEY = os.getenv("AT_MISTRAL_API_KEY", None)

GEMINI_EMBEDDING_MODEL = os.getenv("AT_GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
OPENAI_EMBEDDING_MODEL = os.getenv("AT_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
VOYAGEAI_EMBEDDING_MODEL = os.getenv("AT_VOYAGEAI_EMBEDDING_MODEL", "voyage-large-2")
MISTRAL_EMBEDDING_MODEL = os.getenv("AT_MISTRAL_EMBEDDING_MODEL", "mistral-embed")

LOCAL_EMBEDDING_MODEL = os.getenv("AT_LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
LOCAL_EMBEDDING_DEVICE = os.getenv("AT_LOCAL_EMBEDDING_DEVICE", "auto")

SPEECH_TO_TEXT_PROVIDER = os.getenv("AT_SPEECH_TO_TEXT_PROVIDER", "local")

LOCAL_WHISPER_MODEL = os.getenv("AT_LOCAL_WHISPER_MODEL", "base")
LOCAL_WHISPER_DEVICE = os.getenv("AT_LOCAL_WHISPER_DEVICE", "auto")

LOCAL_OLLAMA_HOST = os.getenv("AT_LOCAL_OLLAMA_HOST", "http://localhost:11434")
LOCAL_OLLAMA_MODEL = os.getenv("AT_LOCAL_OLLAMA_MODEL", "qwen2")

# Initial directory
Path(APP_DIR).mkdir(parents=True, exist_ok=True)
