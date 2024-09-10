import os.path
import uuid
from datetime import timedelta

import pytubefix
import tiktoken
from pytubefix import YouTube, Caption
from pytubefix.cli import on_progress

from engine.assistants import env
from engine.assistants.constants import TEMP_AUDIO_DIR
from engine.assistants.logger import log
from engine.database.models import VideoChapter, Video
from engine.filters.audio_filter import filter_audio
from engine.services.ai_service import AiService
from engine.services.video_service import VideoService


class YoutubeService:
    def __init__(self, url):
        self.__agent = YouTube(
            url=url,
            on_progress_callback=on_progress,
            use_oauth=False,
            allow_oauth_cache=True
        )
        self.__ai_service = AiService()
        self.__video_service = VideoService()

    def fetch_basic_info(self):
        """
        Fetches the basic information of a YouTube video.

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

    async def fetch_video_data(self) -> Video:
        """
        Fetches and processes video data from a YouTube video.

        This function checks if the video data is already available in the database.
        If it is, the function returns the existing video data.
        Otherwise, it creates a new video object, extracts chapters and transcripts,
        and saves the video data to the database.

        Args:
            self: The YoutubeService instance.

        Returns:
            Video: The video data object.
        """
        video = VideoService.find_video_by_youtube_id(self.__agent.video_id)
        if video is not None:
            return video

        video = Video(
            youtube_id=self.__agent.video_id,
            url=self.__agent.watch_url,
            author=self.__agent.author,
            title=self.__agent.title,
            description=self.__agent.description,
            thumbnail=self.__agent.thumbnail_url,
            duration=self.__agent.length
        )
        extracted_chapters = self.__extract_chapters()
        language, extracted_transcripts = self.__extract_transcript()
        video_transcript, video_chapters = self.__pair_video_chapters_with_transcripts(video, extracted_chapters, extracted_transcripts)
        video.amount_chapters = len(video_chapters)
        video.transcript = video_transcript
        video.transcript_tokens = len(tiktoken.get_encoding("cl100k_base").encode(video.transcript))
        video.language = language
        VideoService.save(video, video_chapters)
        return video

    def __extract_chapters(self):
        chapters: list[pytubefix.Chapter] = self.__agent.chapters
        video_chapters: list[VideoChapter] = []
        if chapters is not None and chapters:
            video_chapters.extend(
                VideoChapter(
                    chapter_no=index + 1,
                    title=f"Chapter {index + 1} : {chapter.title} ({timedelta(seconds=chapter.start_seconds)} - {timedelta(seconds=chapter.start_seconds + chapter.duration)})",
                    start_time=chapter.start_seconds,
                    start_label=chapter.start_label,
                    duration=chapter.duration,
                )
                for index, chapter in enumerate(chapters)
            )
            return video_chapters

        # Auto chunk chapter by predefine duration length
        predict_parts = self.__get_predict_chapters_range()
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
                )
            )
        return video_chapters

    def __get_predict_chapters_range(self):
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

    def __get_potential_step(self):
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
        language, audio_file = self.__extract_audio()
        return language, self.__ai_service.speech_to_text(audio_file)

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
                    log.debug("Cannot get start_time or duration")
                    continue
                result.append({
                    'start_time': start_time,
                    'duration': duration,
                    'text': text.replace("\n", " ")
                })
        return result

    def __extract_audio(self):
        log.debug(f"start to download audio {self.__agent.title}")
        output_audio_file = self.__download_audio()

        language = self.__ai_service.recognize_audio_language(
            audio_path=output_audio_file,
            duration=self.__agent.length
        )
        return language, output_audio_file

    def __download_audio(self):
        ys = self.__agent.streams.get_audio_only()
        tmp_audio_file_name = f"{uuid.uuid4()}"
        ys.download(mp3=True, output_path=TEMP_AUDIO_DIR, filename=tmp_audio_file_name, skip_existing=True)
        return filter_audio(os.path.join(TEMP_AUDIO_DIR, f"{tmp_audio_file_name}.mp3"))

    @staticmethod
    def __pair_video_chapters_with_transcripts(video: Video, chapters: list[VideoChapter], transcripts: [{}]):
        sorted_chapters = sorted(chapters, key=lambda chapter: chapter.chapter_no)
        result: list[VideoChapter] = []
        for sorted_chapter in sorted_chapters:
            sorted_chapter.video = video
            start_ms = sorted_chapter.start_time * 1000
            end_ms = (sorted_chapter.start_time + sorted_chapter.duration) * 1000
            chapter_transcript: str = ""
            for transcript in transcripts:
                start_transcript_ms = transcript['start_time']
                duration_transcript_ms = transcript['duration']
                if not start_transcript_ms or not duration_transcript_ms:
                    log.warn("skip this invalid transcript part")
                    continue

                end_transcript_ms = start_transcript_ms + duration_transcript_ms
                if start_transcript_ms < start_ms or end_transcript_ms > end_ms:
                    continue
                chapter_transcript += f"{transcript['text']}\n"
            if chapter_transcript != "":
                log.debug(f"title: {sorted_chapter.title}\ntranscript: {chapter_transcript}")
                sorted_chapter.transcript = chapter_transcript
                result.append(sorted_chapter)
        video_transcript = "\n".join([f"## {x.title}\n-----\n{x.transcript}" for x in result if x.transcript])
        return f"# {video.title}\n-----\n\n{video_transcript}", result
