import warnings

from faster_whisper import WhisperModel

from engine.assistants import env
from engine.services.ai_service import AiService


def download_whisper_model():
    if env.LOCAL_WHISPER_ENABLED:
        warnings.warn("Start downloading model Whisper offline")
        WhisperModel(env.LOCAL_WHISPER_MODEL, device=env.LOCAL_WHISPER_DEVICE, compute_type='int8', download_root=env.APP_DIR)
        warnings.warn("Local model Whisper is ready")


download_whisper_model()
