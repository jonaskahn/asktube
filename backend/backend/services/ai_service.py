import os.path
import random
import tempfile
from collections import Counter
from uuid import uuid4

import google.generativeai as genai
import voyageai
from audio_extract import extract_audio
from faster_whisper import WhisperModel
from future.backports.datetime import timedelta
from openai import OpenAI

from backend import env
from backend.db.specs import chromadb_client
from backend.error.ai_error import AiSegmentError, AiApiKeyError
from backend.utils.logger import log
from backend.utils.prompts import SYSTEM_PROMPT


class AiService:
    def __init__(self):
        pass

    @staticmethod
    def __get_whisper_model():
        compute_type = 'int8' if env.LOCAL_WHISPER_DEVICE == 'cpu' else 'fp16'
        return WhisperModel(env.LOCAL_WHISPER_MODEL, device=env.LOCAL_WHISPER_DEVICE, compute_type=compute_type)

    @staticmethod
    def __get_local_embedding_model():
        local_model_path = os.path.join(env.APP_DIR, env.LOCAL_EMBEDDING_MODEL)
        if not os.path.exists(local_model_path):
            encoder = SentenceTransformer(model, device=env.LOCAL_EMBEDDING_DEVICE, trust_remote_code=True)
            encoder.save(local_model_path)
            return encoder
        return SentenceTransformer(local_model_path, device=env.LOCAL_EMBEDDING_DEVICE, trust_remote_code=True)

    def recognize_audio_language(self, audio_path, duration):
        """
        Recognize language from audio. Random pick a set of split audios (at the end, middle, start) to detect language.
        - If duration is less than 5 minutes, use whole audio. Otherwise, use (30 seconds x 3 ) split audios.
        Note: Current faster-whisper does not support detecting language at the time I write this.
        :param audio_path:
        :param duration:
        :return:
        """
        model = self.__get_whisper_model()
        if duration <= 120:
            _, info = model.transcribe(audio_path)
            return info.language
        start_segment, middle_segment, end_segment = None, None, None
        try:
            start_segment, middle_segment, end_segment = self.__segment_audio(audio_path, duration)
            _, start_info = model.transcribe(start_segment)
            _, middle_info = model.transcribe(middle_segment)
            _, end_info = model.transcribe(end_segment)
            languages = [start_info.language, middle_info.language, end_info.language]
            most_common_lang, count = Counter(languages).most_common(1)[0]
            return most_common_lang if count >= 2 else None
        finally:
            for segment in [start_segment, middle_segment, end_segment]:
                if segment is not None and os.path.exists(segment):
                    os.remove(segment)

    @staticmethod
    def __segment_audio(audio_path, duration):
        if duration < 120:
            raise AiSegmentError("Duration must be greater than 600 seconds")
        start_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")
        extract_audio(
            input_path=audio_path,
            output_path=start_segment_audio_path,
            start_time=f"{timedelta(seconds=0)}",
            duration=env.AUDIO_CHUNK_SHORT_DURATION
        )

        middle_start = random.randint(
            duration // env.AUDIO_CHUNK_SHORT_DURATION,
            duration // 3 - env.AUDIO_CHUNK_SHORT_DURATION
        )
        middle_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")

        extract_audio(
            input_path=audio_path,
            output_path=middle_segment_audio_path,
            start_time=f"{timedelta(seconds=middle_start)}",
            duration=env.AUDIO_CHUNK_SHORT_DURATION
        )

        end_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")
        extract_audio(
            input_path=audio_path,
            output_path=end_segment_audio_path,
            start_time=f"{timedelta(seconds=duration - env.AUDIO_CHUNK_SHORT_DURATION * 2)}"
        )

        return start_segment_audio_path, middle_segment_audio_path, end_segment_audio_path

    def speech_to_text(self, audio_path):
        model = self.__get_whisper_model()
        segments, _ = model.transcribe(audio=audio_path, beam_size=5)
        return [
            {
                'start_time': int(segment.start * 1000.0),
                'duration': int((segment.end - segment.start) * 1000.0),
                'text': segment.text,
            }
            for segment in segments
        ]

    @staticmethod
    def embedding_document_with_gemini(text: str):
        if env.GEMINI_API_KEY is None or env.GEMINI_API_KEY.strip() == "":
            raise AiApiKeyError()
        try:
            genai.configure(api_key=env.GEMINI_API_KEY)
            result = genai.embed_content(model=env.GEMINI_EMBEDDING_MODEL, content=text)
            return result['embedding']
        except Exception as e:
            log.debug(f"\nError in embedding_document_with_gemini: \n{text}", exc_info=True)
            raise e

    @staticmethod
    def embedding_document_with_openai(text: str):
        if env.OPENAI_API_KEY is None or env.OPENAI_API_KEY.strip() == "":
            raise AiApiKeyError()
        client = OpenAI(api_key=env.OPENAI_API_KEY)
        result = client.embeddings.create(model=env.OPENAI_EMBEDDING_MODEL, input=[text])
        return result.data[0].embedding

    @staticmethod
    def embedding_document_with_voyageai(text: str):
        if env.VOYAGEAI_API_KEY is None or env.VOYAGEAI_API_KEY.strip() == "":
            raise AiApiKeyError()
        client = voyageai.Client(api_key=env.VOYAGEAI_API_KEY)
        result = client.embed(texts=[text], model=env.VOYAGEAI_EMBEDDING_MODEL, input_type="document")
        return result.embeddings[0]

    @staticmethod
    def embedding_document_with_local(text: str):
        encoder = AiService.__get_local_embedding_model()
        embeddings = encoder.encode([text])
        return embeddings[0]

    @staticmethod
    def store_embeddings(table: str, ids: list[str], texts: list[str], embeddings: list[list[float]]):
        collection = chromadb_client.get_or_create_collection(table)
        collection.add(ids=ids, embeddings=embeddings, documents=texts)

    @staticmethod
    def query_embeddings(table: str, query: list[list[float]], fetch_size: int = 5, expect_size: int = 2, max_distance: float = 1.2):
        collection = chromadb_client.get_or_create_collection(table)
        results = collection.query(query_embeddings=query, n_results=fetch_size, include=['documents', 'distances'])

        distances = results['distances']
        documents = results['documents']
        flat_distances = [dist for sublist in distances for dist in sublist]
        flat_documents = [doc for sublist in documents for doc in sublist]

        distance_doc_pairs = list(zip(flat_distances, flat_documents))
        filtered_pairs = [pair for pair in distance_doc_pairs if pair[0] <= max_distance]
        sorted_pairs = sorted(filtered_pairs, key=lambda pair: pair[0])
        top_closest = sorted_pairs[:max(1, expect_size)]

        unique_documents = []
        seen_docs = set()
        for _, doc in top_closest:
            doc_id = id(doc)
            if doc_id not in seen_docs:
                unique_documents.append(doc)
                seen_docs.add(doc_id)
        return "\n".join(unique_documents)

    @staticmethod
    def generate_text_with_openai(
            model: str,
            prompt: str,
            max_tokens: int = 2048,
            temperature: float = 0.7,
            top_p: float = 1.0
    ):
        if env.OPENAI_API_KEY is None or env.OPENAI_API_KEY.strip() == "":
            raise AiApiKeyError()
        client = OpenAI(api_key=env.OPENAI_API_KEY)
        return ""

    @staticmethod
    def generate_text_with_gemini(
            prompt: str,
            model: str,
            max_tokens: int = 4096,
            temperature: float = 0.6,
            top_p: float = 0.6,
            top_k: int = 32
    ):
        if env.GEMINI_API_KEY is None or env.GEMINI_API_KEY.strip() == "":
            raise AiApiKeyError()
        genai.configure(api_key=env.GEMINI_API_KEY)
        agent = genai.GenerativeModel(model_name=model if model is not None else "gemini-1.5-flash", system_instruction=SYSTEM_PROMPT)
        response = agent.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        )
        return response.text.replace("\n", " ").strip()

    @staticmethod
    def chat_with_openai(
            model: str,
            prompt: str,
            previous_chats: list,
            max_tokens: int = 2048,
            temperature: float = 0.7,
            top_p: float = 1.0
    ):
        if env.OPENAI_API_KEY is None or env.OPENAI_API_KEY.strip() == "":
            raise AiApiKeyError()
        client = OpenAI(api_key=env.OPENAI_API_KEY)
        return ""

    @staticmethod
    def chat_with_gemini(
            prompt: str,
            model: str,
            previous_chats: list,
            max_tokens: int = 4096,
            temperature: float = 0.6,
            top_p: float = 0.6,
            top_k: int = 32
    ):
        if env.GEMINI_API_KEY is None or env.GEMINI_API_KEY.strip() == "":
            raise AiApiKeyError()
        genai.configure(api_key=env.GEMINI_API_KEY)
        agent = genai.GenerativeModel(model_name=model if model is not None else "gemini-1.5-flash", system_instruction=SYSTEM_PROMPT)
        chat = agent.start_chat(history=previous_chats)
        response = chat.send_message(prompt)
        return response.text
