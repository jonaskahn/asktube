from backend.db.models import Video, VideoChapter
from backend.db.specs import sqlite_client
from backend.error.video_error import VideoNotFoundError
from backend.services.ai_service import AiService


class VideoService:
    def __init__(self):
        self.__ai_service = AiService()

    @staticmethod
    def save(video: Video, chapters: list[VideoChapter]):
        with sqlite_client.atomic() as transaction:
            try:
                video.save()
                for chapter in chapters:
                    chapter.save()
                transaction.commit()
            except Exception as e:
                transaction.rollback()
                raise e

    @staticmethod
    def find_video_by_youtube_id(youtube_id: str):
        return Video.get_or_none(Video.youtube_id == youtube_id)

    @staticmethod
    def find_video_by_id(vid: int):
        return Video.get_or_none(Video.id == vid)

    async def analysis_video(self, vid: int):
        video = self.find_video_by_id(vid)
        if video is None:
            raise VideoNotFoundError("Video not found")
        video_chapters = list(VideoChapter.select().where(VideoChapter.video == video))
        return
