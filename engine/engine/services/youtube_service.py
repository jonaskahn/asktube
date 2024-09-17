import json
import os.path
import uuid
from datetime import timedelta
from pathlib import Path

import pytubefix
from audio_extract import extract_audio
from playhouse.shortcuts import model_to_dict
from pytubefix import YouTube, Caption
from pytubefix.cli import on_progress
from sanic.log import logger

from engine.database.models import VideoChapter, Video
from engine.processors.audio_processor import process_audio
from engine.services.ai_service import AiService
from engine.services.video_service import VideoService
from engine.supports import env
from engine.supports.constants import TEMP_AUDIO_DIR


class YoutubeService:
    def __init__(self, url):
        self.__agent = YouTube(
            url=url,
            on_progress_callback=on_progress,
            use_oauth=False,
            allow_oauth_cache=True
        )

    def fetch_basic_info(self):
        """
        Fetches and returns the basic information of a YouTube video.

        Returns:
            dict: A dictionary containing the title, description, duration, author, thumbnail, and captions of the video.
        """

        captions = list(map(lambda c: {'name': c.name, 'value': c.code}, self.__agent.captions))
        return {
            'title': self.__agent.title,
            'description': self.__agent.description,
            'duration': str(timedelta(seconds=self.__agent.length)),
            'author': self.__agent.author,
            'thumbnail': self.__agent.thumbnail_url,
            'captions': captions
        }

    async def fetch_video_data(self, provider: str) -> Video:

        video = VideoService.find_video_by_youtube_id(self.__agent.video_id)
        if video is not None:
            return video

        video = Video(
            youtube_id=self.__agent.video_id,
            url=self.__agent.watch_url,
            play_url=self.__agent.embed_url,
            author=self.__agent.author,
            title=self.__agent.title,
            description=self.__agent.description,
            thumbnail=self.__agent.thumbnail_url,
            duration=self.__agent.length,
            embedding_provider=provider
        )
        video_chapters = self.__extract_chapters()
        language, transcript = self.__extract_transcript()
        video.language = language
        video.raw_transcript = json.dumps(transcript, ensure_ascii=False) if transcript else None
        video.amount_chapters = len(video_chapters)
        VideoService.save(video, video_chapters)
        return model_to_dict(video)

    def __extract_chapters(self) -> list[VideoChapter]:
        chapters: list[pytubefix.Chapter] = self.__agent.chapters
        video_chapters: list[VideoChapter] = []
        if chapters is not None and chapters:
            video_chapters.extend(
                VideoChapter(
                    chapter_no=index + 1,
                    title=f"Chapter {index + 1} : {chapter.title} ({timedelta(seconds=chapter.start_seconds)} - {timedelta(seconds=chapter.start_seconds + chapter.duration)})",
                    start_time=chapter.start_seconds,
                    start_label=chapter.start_label,
                    duration=chapter.duration
                )
                for index, chapter in enumerate(chapters)
            )
            return video_chapters

        # Auto chunk chapter by predefine duration length
        predict_parts = self.__get_predict_chapters_range()
        audio_path_file = self.__download_audio()
        has_captions = self.__agent.captions is not None and len(self.__agent.captions) != 0
        for index, _ in enumerate(predict_parts):
            if len(predict_parts) == index + 1:
                break

            current_start_seconds = predict_parts[index]
            next_start_seconds = predict_parts[index + 1]
            video_chapters.append(
                VideoChapter(
                    chapter_no=index + 1,
                    title=f"Chapter {index + 1} ({timedelta(seconds=current_start_seconds)} - {timedelta(seconds=next_start_seconds)})",
                    start_time=current_start_seconds,
                    start_label=str(timedelta(seconds=current_start_seconds)),
                    duration=(next_start_seconds - current_start_seconds),
                    audio_path=self.__chunk_audio_task(
                        has_captions=has_captions,
                        audio_path_file=audio_path_file,
                        start_time=str(timedelta(seconds=current_start_seconds)),
                        duration=(next_start_seconds - current_start_seconds - 10)
                    )
                )
            )
        if Path(audio_path_file).exists():
            os.remove(audio_path_file)
        return video_chapters

    @staticmethod
    def __chunk_audio_task(has_captions: bool, audio_path_file: str, start_time: str, duration: int) -> str | None:
        if has_captions:
            return None
        output_path = os.path.join(TEMP_AUDIO_DIR, f"{uuid.uuid4()}.wav")
        extract_audio(
            input_path=audio_path_file,
            output_path=output_path,
            start_time=start_time,
            duration=duration,
            output_format="wav",
            overwrite=True
        )
        return output_path

    def __get_predict_chapters_range(self) -> list[int]:
        duration = self.__agent.length
        if duration <= 0:
            return []
        step = self.__get_potential_step()
        if step >= duration:
            return [0, duration]
        predict_parts = list(range(0, duration, step))
        if duration not in predict_parts:
            predict_parts.append(duration)
        return predict_parts

    def __get_potential_step(self) -> int:
        default_chunk_step = env.AUDIO_CHUNK_CHAPTER_DURATION
        if default_chunk_step is not None:
            return int(default_chunk_step)
        return min(self.__agent.length, 600)

    def __extract_transcript(self):
        transcripts = self.__agent.caption_tracks
        if transcripts is not None and transcripts:
            prefer_lang = env.LANGUAGE_PREFER_USAGE
            for transcript in transcripts:
                if prefer_lang in transcript.code:
                    return transcript.code.replace("a.", ""), self.__combine_youtube_caption_data(transcript)
            transcript = transcripts[0]
            return transcript.code.replace("a.", ""), self.__combine_youtube_caption_data(transcript)
        return None, []

    @staticmethod
    def __combine_youtube_caption_data(caption: Caption):
        events = caption.json_captions['events']
        result = []
        for e in events:
            subs = e.get('segs', [])
            if len(subs) == 0:
                continue
            text = ''.join(item.get('utf8', '') for item in subs)
            if text is not None and text.strip() != '':
                start_time = e.get('tStartMs')
                duration = e.get('dDurationMs')
                if start_time is None or duration is None:
                    logger.debug("Cannot get start_time or duration")
                    continue
                result.append({
                    'start_time': start_time,
                    'duration': duration,
                    'text': text.replace("\n", " ")
                })
        return result

    def __extract_audio(self):
        logger.debug(f"start to download audio {self.__agent.title}")
        output_audio_file = self.__download_audio()
        language = AiService.recognize_audio_language(
            audio_path=output_audio_file,
            duration=self.__agent.length
        )
        return language, output_audio_file

    def __download_audio(self) -> str:
        ys = self.__agent.streams.get_audio_only()
        tmp_audio_file_name = f"{uuid.uuid4()}"
        ys.download(mp3=True, output_path=TEMP_AUDIO_DIR, filename=tmp_audio_file_name, skip_existing=True)
        return process_audio(os.path.join(TEMP_AUDIO_DIR, f"{tmp_audio_file_name}.mp3"))
