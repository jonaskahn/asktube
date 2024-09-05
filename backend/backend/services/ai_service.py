import os.path
import random
import tempfile
from collections import Counter
from uuid import uuid4

import anthropic
import google.generativeai as genai
import tiktoken
import voyageai
from audio_extract import extract_audio
from faster_whisper import WhisperModel
from future.backports.datetime import timedelta
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from backend import env
from backend.db.models import Chat
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
    def __get_local_embedding_encoder():
        local_model_path: str = os.path.join(env.APP_DIR, env.LOCAL_EMBEDDING_MODEL)
        if not os.path.exists(local_model_path):
            encoder = SentenceTransformer(model_name_or_path=env.LOCAL_EMBEDDING_MODEL, device=env.LOCAL_EMBEDDING_DEVICE, trust_remote_code=True)
            encoder.save(local_model_path)
            return encoder
        return SentenceTransformer(model_name_or_path=local_model_path, device=env.LOCAL_EMBEDDING_DEVICE, trust_remote_code=True)

    def recognize_audio_language(self, audio_path, duration):
        """
        Recognize language from the audio path. 
        - If audio length is less than 120 seconds, use whole audio to detecht
        - Otherwise, random pick a set of split audios (at the end, middle, start) to detect language.
        Note: Current faster-whisper does not support any "fast way" detecting languages. Here is my work-around solution.
        
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
    def __chunk_text(text, max_tokens):
        encoding = tiktoken.get_encoding("cl100k_base")
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_tokens = encoding.encode(word)
            if current_length + len(word_tokens) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word_tokens)
            else:
                current_chunk.append(word)
                current_length += len(word_tokens)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def embedding_document_with_gemini(text: str, max_tokens=2000):
        if env.GEMINI_API_KEY is None or env.GEMINI_API_KEY.strip() == "":
            raise AiApiKeyError()
        try:
            texts = AiService.__chunk_text(text, max_tokens)
            genai.configure(api_key=env.GEMINI_API_KEY)
            return texts, [genai.embed_content(model=env.GEMINI_EMBEDDING_MODEL, content=text)['embedding'] for text in texts]
        except Exception as e:
            log.debug(f"\nError in embedding_document_with_gemini: \n{text}", exc_info=True)
            raise e

    @staticmethod
    def embedding_document_with_openai(text: str, max_tokens=8000):
        if env.OPENAI_API_KEY is None or env.OPENAI_API_KEY.strip() == "":
            raise AiApiKeyError()
        texts = AiService.__chunk_text(text, max_tokens)
        client = OpenAI(api_key=env.OPENAI_API_KEY)
        return texts, [client.embeddings.create(model=env.OPENAI_EMBEDDING_MODEL, input=[text]).data[0].embedding for text in texts]

    @staticmethod
    def embedding_document_with_voyageai(text: str, max_tokens=16000):
        if env.VOYAGEAI_API_KEY is None or env.VOYAGEAI_API_KEY.strip() == "":
            raise AiApiKeyError()
        texts = AiService.__chunk_text(text, max_tokens)
        client = voyageai.Client(api_key=env.VOYAGEAI_API_KEY)
        return texts, [client.embed(texts=[text], model=env.VOYAGEAI_EMBEDDING_MODEL, input_type="document").embeddings[0] for text in texts]

    @staticmethod
    def embedding_document_with_local(text: str, max_tokens=500):
        texts = AiService.__chunk_text(text, max_tokens)
        encoder = AiService.__get_local_embedding_encoder()
        return texts, [encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True).tolist()[0] for text in texts]

    @staticmethod
    def store_embeddings(table: str, ids: list[str], texts: list[str], embeddings: list[list[float]]):
        collection = chromadb_client.get_or_create_collection(table)
        collection.add(ids=ids, embeddings=embeddings, documents=texts)

    @staticmethod
    def query_embeddings(table: str, query: list[list[float]], fetch_size: int = 10, thresholds: list[float] = None):
        if thresholds is None:
            thresholds = [0.3, 0.6]
        collection = chromadb_client.get_or_create_collection(table)
        results = collection.query(query_embeddings=query, n_results=fetch_size, include=['documents', 'distances'])

        distances = results['distances']
        documents = results['documents']
        flat_distances = [dist for sublist in distances for dist in sublist]
        flat_documents = [doc for sublist in documents for doc in sublist]

        distance_doc_pairs = list(zip(flat_distances, flat_documents))
        top_closest = []
        for threshold in thresholds:
            filtered_pairs = [pair for pair in distance_doc_pairs if pair[0] <= threshold]
            sorted_pairs = sorted(filtered_pairs, key=lambda pair: pair[0])
            top_closest.extend(sorted_pairs[:max(1, fetch_size)])

        unique_documents = []
        seen_docs = set()
        for _, doc in top_closest:
            doc_id = id(doc)
            if doc_id not in seen_docs:
                unique_documents.append(doc)
                seen_docs.add(doc_id)
        return len(unique_documents), "\n".join(unique_documents)

    @staticmethod
    def chat_with_gemini(
            model: str,
            prompt: str,
            system_prompt: str = SYSTEM_PROMPT,
            previous_chats: list[Chat] = None,
            max_tokens: int = 4096,
            temperature: float = 0.6,
            top_p: float = 0.6,
            top_k: int = 32
    ):
        if previous_chats is None:
            previous_chats = []
        if env.GEMINI_API_KEY is None or env.GEMINI_API_KEY.strip() == "":
            raise AiApiKeyError("Refine your Gemini API key in the .env file")
        chat_histories = AiService.__build_gemini_chat_history(previous_chats)
        genai.configure(api_key=env.GEMINI_API_KEY)
        agent = genai.GenerativeModel(
            model_name=model if model is not None or model else "gemini-1.5-flash",
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        )
        chat = agent.start_chat(history=chat_histories)
        response = chat.send_message(prompt)
        return response.text.removesuffix("\n").strip()

    @staticmethod
    def __build_gemini_chat_history(chats: list[Chat]):
        if chats is None or not chats:
            return []
        chat_histories = []
        for chat in chats:
            chat_histories.extend(
                (
                    {"role": "user", "parts": chat.refined_question},
                    {"role": "model", "parts": chat.answer},
                )
            )
        return chat_histories

    @staticmethod
    def chat_with_openai(
            model: str,
            prompt: str,
            system_prompt: str = SYSTEM_PROMPT,
            previous_chats: list[Chat] = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            top_p: float = 0.8
    ):
        if previous_chats is None:
            previous_chats = []
        if env.OPENAI_API_KEY is None or env.OPENAI_API_KEY.strip() == "":
            raise AiApiKeyError("Refine your OpenAI API key in the .env file")
        client = OpenAI(api_key=env.OPENAI_API_KEY)
        messages = AiService.__build_openai_chat_history(system_prompt, previous_chats)
        messages.append({
            "role": "user",
            "content": prompt
        })
        completion = client.chat.completions.create(
            model=model if model is not None or model else "gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return completion.choices[0].message.content

    @staticmethod
    def __build_openai_chat_history(system_prompt: str, chats: list[Chat]):
        if chats is None or not chats:
            return []
        chat_histories = [{
            "role": "system",
            "content": system_prompt
        }]
        for chat in chats:
            chat_histories.extend(
                (
                    {"role": "user", "content": chat.refined_question},
                    {"role": "assistant", "content": chat.answer},
                )
            )
        return chat_histories

    @staticmethod
    def chat_with_claude(
            model: str,
            prompt: str,
            system_prompt: str = SYSTEM_PROMPT,
            previous_chats: list[Chat] = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            top_p: float = 0.7,
            top_k: int = 16
    ):
        if previous_chats is None:
            previous_chats = []
        if env.CLAUDE_API_KEY is None or env.CLAUDE_API_KEY.strip() == "":
            raise AiApiKeyError("Refine your Claude API key in the .env file")
        client = anthropic.Anthropic(api_key=env.CLAUDE_API_KEY)
        messages = AiService.__build_claude_chat_history(chats=previous_chats)
        messages.append({
            "role": "user", "content": prompt
        })
        response = client.messages.create(
            model=model if model is not None or model else "claude-3-haiku-20240307",
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        return response.content[0].text

    @staticmethod
    def __build_claude_chat_history(chats: list[Chat]):
        if chats is None or not chats:
            return []
        chat_histories = []
        for chat in chats:
            chat_histories.extend(
                (
                    {"role": "user", "content": chat.refined_question},
                    {"role": "assistant", "content": chat.answer},
                )
            )
        return chat_histories

    @staticmethod
    def chat_with_ollama(
            model: str,
            prompt: str,
            system_prompt: str = SYSTEM_PROMPT,
            previous_chats: list[Chat] = None,
            max_tokens: int = 2048,
            temperature: float = 0.7,
            top_p: float = 1.0
    ):
        if previous_chats is None:
            previous_chats = []
        raise NotImplementedError("OLLAMA is not implemented")
