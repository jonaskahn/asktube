import os
from typing import Callable, List
from uuid import uuid4

import noisereduce as nr
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile

from engine.assistants.env import APP_DIR
from engine.assistants.logger import log

temp_audio_dir = os.path.join(APP_DIR, "temp")
os.makedirs(temp_audio_dir, exist_ok=True)


class __AudioChainFilter:
    def __init__(self):
        self.filters: List[Callable[[str], str]] = []

    def add_filter(self, func: Callable[[str], str]) -> '__AudioChainFilter':
        self.filters.append(func)
        return self

    def filter(self, audio_input_path: str) -> str:
        for fnc in self.filters:
            original_audio_file_path = audio_input_path
            try:
                log.debug(f"audio input: {audio_input_path}")
                audio_input_path = fnc(audio_input_path)
                log.debug(f"audio output: {audio_input_path}")
            except Exception as e:
                raise e
            finally:
                os.remove(original_audio_file_path)
        return audio_input_path


def mp4_to_wav(audio_input_path: str) -> str:
    audio_output_file = os.path.join(temp_audio_dir, f"{uuid4()}.wav")
    sound = AudioSegment.from_file(audio_input_path, format="mp4")
    sound.export(audio_output_file, format="wav")
    return audio_output_file


def denoise(audio_path: str) -> str:
    rate, data = wavfile.read(audio_path)
    if data.ndim == 1:
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
    else:
        channels = []
        for i in range(data.shape[1]):
            raw_data = data[:, i]
            reduced_noise_channel = nr.reduce_noise(y=raw_data, sr=rate)
            channels.append(reduced_noise_channel)
        reduced_noise = np.stack(channels, axis=1)
    audio_output_file = os.path.join(temp_audio_dir, f"{uuid4()}.wav")
    wavfile.write(audio_output_file, rate, reduced_noise.astype(np.int16))
    return audio_output_file


def filter_audio(audio_path: str) -> str:
    return __AudioChainFilter().add_filter(mp4_to_wav).add_filter(denoise).filter(audio_path)
