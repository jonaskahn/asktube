import os.path
from datetime import timedelta
from pathlib import Path

import pytubefix
from pytubefix import YouTube, Caption
from pytubefix.cli import on_progress

from backend.constants import YT_AUDIO_ABS_FILE_NAME, YT_AUDIO_FILE_NAME
from backend.db.models import VideoChapter
from backend.env import APP_DIR, VIDEO_CHUNK_LENGTH
from backend.services.ai_service import AiService
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

    def fetch_video_data(self):
        video_chapters = self.__extract_chapters()
        raw_transcript = self.__extract_transcript()
        log.debug("Extracted transcript: %s", raw_transcript)

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
        # transcripts = self.__agent.caption_tracks
        # if transcripts is not None and transcripts:
        #     for transcript in transcripts:
        #         if LANGUAGE_PREFER_USAGE in transcript.code:
        #             return transcript.code.replace("a.", ""), self.__combine_youtube_caption_data(transcript)
        #     transcript = transcripts[0]
        #     return transcript.code, self.__combine_youtube_caption_data(transcript)
        language, audio_file = self.__download_audio()
        return self.__ai_service.speech_to_text(audio_file)

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

        log.info(f"Finished download audio {self.__agent.title} to {output_audio_dir}")

        language = self.__ai_service.recognize_audio_language(
            audio_path=os.path.join(output_audio_dir, YT_AUDIO_ABS_FILE_NAME),
            duration=self.__agent.length
        )
        return language, os.path.join(output_audio_dir, YT_AUDIO_ABS_FILE_NAME)
