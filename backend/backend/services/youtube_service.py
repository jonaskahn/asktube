import os.path
from datetime import timedelta
from pathlib import Path

import pytubefix
from pytubefix import YouTube, Caption
from pytubefix.cli import on_progress

from backend.constants import YT_AUDIO_ABS_FILE_NAME, YT_AUDIO_FILE_NAME
from backend.db.models import VideoChapter, Video
from backend.env import APP_DIR, VIDEO_CHUNK_LENGTH, LANGUAGE_PREFER_USAGE
from backend.services.ai_service import AiService
from backend.services.video_service import VideoService
from backend.utils.logger import log


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
        captions = list(map(lambda c: {'name': c.name, 'value': c.code}, self.__agent.captions))
        return {
            'title': self.__agent.title,
            'description': self.__agent.description,
            'duration': str(timedelta(seconds=self.__agent.length)),
            'author': self.__agent.author,
            'thumbnail': self.__agent.thumbnail_url,
            'captions': captions
        }

    async def fetch_video_data(self):
        video = self.__video_service.find_video_by_youtube_id(self.__agent.video_id)
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
        video_transcript, video_chapters = self.__paring_video_chapters(video, extracted_chapters, extracted_transcripts)
        video.amount_chapters = len(video_chapters)
        video.transcript = video_transcript
        video.language = language
        self.__video_service.save(video, video_chapters)
        return video

    def __extract_chapters(self):
        chapters: list[pytubefix.Chapter] = self.__agent.chapters
        video_chapters: list[VideoChapter] = []
        if chapters is not None and chapters:
            video_chapters.extend(
                VideoChapter(
                    chapter_no=index + 1,
                    title=chapter.title,
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
                    title=f"Part {index} ({timedelta(seconds=current_start_seconds)} - {timedelta(seconds=next_start_seconds)})",
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
        default_chunk_step = VIDEO_CHUNK_LENGTH
        if default_chunk_step is not None:
            return int(default_chunk_step)
        return min(self.__agent.length, 600)

    def __extract_transcript(self):
        transcripts = self.__agent.caption_tracks
        prefer_lang = LANGUAGE_PREFER_USAGE
        if transcripts is not None and transcripts:
            for transcript in transcripts:
                if prefer_lang in transcript.code:
                    return transcript.code.replace("a.", ""), self.__combine_youtube_caption_data(transcript)
            transcript = transcripts[0]
            return transcript.code.replace("a.", ""), self.__combine_youtube_caption_data(transcript)
        language, audio_file = self.__download_audio()
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

    def __download_audio(self):
        log.debug(f"Start to download audio {self.__agent.title}")

        output_audio_dir = os.path.join(APP_DIR, f"{self.__agent.video_id}")
        Path(output_audio_dir).mkdir(parents=True, exist_ok=True)

        ys = self.__agent.streams.get_audio_only()
        ys.download(mp3=True, output_path=output_audio_dir, filename=YT_AUDIO_FILE_NAME, skip_existing=True)
        output_audio_file = os.path.join(output_audio_dir, YT_AUDIO_ABS_FILE_NAME)
        log.info(f"Finished download audio {self.__agent.title} to {output_audio_file}")

        language = self.__ai_service.recognize_audio_language(
            audio_path=os.path.join(output_audio_dir, YT_AUDIO_ABS_FILE_NAME),
            duration=self.__agent.length
        )
        return language, output_audio_file

    @staticmethod
    def __paring_video_chapters(video: Video, chapters: list[VideoChapter], transcripts: [{}]):
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
                log.debug(f"title: {sorted_chapter.title}\nno: {sorted_chapter.chapter_no}\ntranscript: {chapter_transcript}")
                sorted_chapter.transcript = chapter_transcript
                result.append(sorted_chapter)
        video_transcript = "\n".join([f"## {x.title}\n-----\n{x.transcript}" for x in result if x.transcript])
        return f"# {video.title}\n-----\n\n{video_transcript}", result
