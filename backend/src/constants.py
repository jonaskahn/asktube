import os

from backend.src.env import APP_DIR

YT_SUB_FILE_NAME: str = "subtitles.srt"
YT_AUDIO_FILE_NAME: str = "audio"
YT_AUDIO_FILE_FORMAT: str = "mp3"

VT_VIDEO_TABLE_TEMPLATE: str = "video_chapters_{id}_{lang}"

SQL_DATABASE = os.path.join(APP_DIR, "asktube.sqlite3")
VECTOR_DATABASE = os.path.join(APP_DIR, "vector")
