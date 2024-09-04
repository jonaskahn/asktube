import os.path
import random
import tempfile
from collections import Counter
from uuid import uuid4

from audio_extract import extract_audio
from faster_whisper import WhisperModel
from future.backports.datetime import timedelta

from backend.env import WHISPER_DEVICE, WHISPER_MODEL, LANGUAGE_DETECT_SEGMENT_LENGTH
from backend.error.ai_service_error import AiSegmentError


class AiService:
    def __init__(self):
        pass

    @staticmethod
    def __get_whisper_model():
        compute_type = 'int8' if WHISPER_DEVICE == 'cpu' else 'fp16'
        return WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=compute_type)

    def recognize_audio_language(self, audio_path, duration):
        """
        Recognize language from audio. Random pick a set of split audios (at the end, middle, start) to detect language.
        - If duration is less than 5 minutes, use whole audio. Otherwise, use (30 seconds x 3 ) split audios.
        Note: Current faster-whisper does not support detecting language at the time I write this.
        :param audio_path:
        :param duration:
        :return:
        """
        start_segment, middle_segment, end_segment = self.__segment_audio(audio_path, duration)
        try:
            model = self.__get_whisper_model()
            if duration <= 600:
                _, info = model.transcribe(audio_path)
                return info.language
            _, start_info = model.transcribe(start_segment)
            _, middle_info = model.transcribe(middle_segment)
            _, end_info = model.transcribe(end_segment)
            languages = [start_info.language, middle_info.language, end_info.language]
            most_common_lang, count = Counter(languages).most_common(1)[0]
            return most_common_lang if count >= 2 else None
        finally:
            for segment in [start_segment, middle_segment, end_segment]:
                if os.path.exists(segment):
                    os.remove(segment)

    @staticmethod
    def __segment_audio(audio_path, duration):
        if duration < 600:
            raise AiSegmentError("Duration must be greater than 600 seconds")
        start_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")
        extract_audio(
            input_path=audio_path,
            output_path=start_segment_audio_path,
            start_time=f"{timedelta(seconds=0)}",
            duration=LANGUAGE_DETECT_SEGMENT_LENGTH
        )

        middle_start = random.randint(
            duration // LANGUAGE_DETECT_SEGMENT_LENGTH,
            duration // 3 - LANGUAGE_DETECT_SEGMENT_LENGTH
        )
        middle_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")

        extract_audio(
            input_path=audio_path,
            output_path=middle_segment_audio_path,
            start_time=f"{timedelta(seconds=middle_start)}",
            duration=LANGUAGE_DETECT_SEGMENT_LENGTH
        )

        end_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")
        extract_audio(
            input_path=audio_path,
            output_path=end_segment_audio_path,
            start_time=f"{timedelta(seconds=duration - LANGUAGE_DETECT_SEGMENT_LENGTH * 2)}"
        )

        return start_segment_audio_path, middle_segment_audio_path, end_segment_audio_path

    def speech_to_text(self, audio_path):
        model = self.__get_whisper_model()
        segments, _ = model.transcribe(audio=audio_path, beam_size=10)
        return [
            {
                'start_time': segment.start,
                'duration': segment.end - segment.start,
                'text': segment.text,
            }
            for segment in segments
        ]
