import os

from engine.supports.env import APP_DIR

YT_AUDIO_FILE_NAME: str = "audio"
YT_AUDIO_FILE_FORMAT: str = "mp3"
YT_AUDIO_ABS_FILE_NAME: str = f"{YT_AUDIO_FILE_NAME}.{YT_AUDIO_FILE_FORMAT}"

VT_VIDEO_TABLE_TEMPLATE: str = "video_chapters_{id}_{lang}"

SQL_DATABASE = os.path.join(APP_DIR, "asktube.sqlite3")
VECTOR_DATABASE = os.path.join(APP_DIR, "vector")

ANALYSIS_STAGE_INITIAL = 0
ANALYSIS_STAGE_COMPLETED = 1
ANALYSIS_STAGE_PROCESSING = 2

TEMP_AUDIO_DIR = os.path.join(APP_DIR, "temp-audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
