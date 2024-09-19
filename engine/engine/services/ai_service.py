import os.path
import random
import tempfile
from collections import Counter
from uuid import uuid4

import anthropic
import google.generativeai as genai
import tiktoken
import torch
import voyageai
from audio_extract import extract_audio
from faster_whisper import WhisperModel
from future.backports.datetime import timedelta
from mistralai import Mistral
from ollama import Client
from openai import OpenAI
from sanic.log import logger
from sentence_transformers import SentenceTransformer

from engine.database.specs import chromadb_client
from engine.supports import env
from engine.supports.errors import AiError
from engine.supports.prompts import SYSTEM_PROMPT
from engine.supports.utils import sha256

has_cuda = torch.cuda.is_available()
has_mps = torch.backends.mps.is_available()


class AiService:
    """
    A service class that provides various AI-related functionalities including speech recognition,
    text embedding, and interaction with different AI chat models.
    """

    def __init__(self):
        pass

    @staticmethod
    def __get_local_whisper_model() -> WhisperModel:
        compute_type = 'fp16' if has_cuda else 'int8'
        return WhisperModel(
            env.LOCAL_WHISPER_MODEL,
            device=env.LOCAL_WHISPER_DEVICE,
            compute_type=compute_type,
            download_root=env.APP_DIR
        )

    @staticmethod
    def __get_local_embedding_encoder() -> SentenceTransformer:
        if has_mps:
            device = "mps"
        elif has_cuda:
            device = "cuda"
        else:
            device = "cpu"

        logger.debug(f"using {device} for embedding")
        local_model_path: str = str(os.path.join(env.APP_DIR, env.LOCAL_EMBEDDING_MODEL))
        if not os.path.exists(local_model_path):
            encoder = SentenceTransformer(
                model_name_or_path=env.LOCAL_EMBEDDING_MODEL,
                device=device if env.LOCAL_EMBEDDING_DEVICE == "auto" else env.LOCAL_EMBEDDING_DEVICE,
                trust_remote_code=True
            )
            encoder.save(local_model_path)
            return encoder
        return SentenceTransformer(model_name_or_path=local_model_path, device=device, trust_remote_code=True)

    @staticmethod
    def recognize_audio_language(audio_path: str, duration: int) -> str:
        """
        Recognizes the language of the audio file.

        This method uses a local Whisper model to transcribe a portion of the audio and determine its language.
        For longer audio files, it samples from the beginning, middle, and end to make a more accurate determination.

        Args:
            audio_path (str): The path to the audio file.
            duration (int): The duration of the audio file in seconds.

        Returns:
            str: The recognized language code.

        Raises:
            AiError: If there's an issue with audio processing or language recognition.
        """
        logger.debug("start to recognize audio language")
        model = AiService.__get_local_whisper_model()
        if duration <= env.AUDIO_CHUNK_RECOGNIZE_THRESHOLD:
            _, info = model.transcribe(audio_path)
            return info.language
        start_segment, middle_segment, end_segment = None, None, None
        try:
            start_segment, middle_segment, end_segment = AiService.__split_segment_audio(audio_path, duration)
            _, start_info = model.transcribe(start_segment)
            _, middle_info = model.transcribe(middle_segment)
            _, end_info = model.transcribe(end_segment)
            languages = [start_info.language, middle_info.language, end_info.language]
            most_common_lang, count = Counter(languages).most_common(1)[0]
            return most_common_lang if count >= 2 else None
        finally:
            logger.debug("finish to recognize audio language")
            for segment in [start_segment, middle_segment, end_segment]:
                if segment is not None and os.path.exists(segment):
                    os.remove(segment)

    @staticmethod
    def __split_segment_audio(audio_path: str, duration: int) -> tuple[str, str, str]:
        if duration < env.AUDIO_CHUNK_RECOGNIZE_THRESHOLD:
            raise AiError(f"duration must be greater than {env.AUDIO_CHUNK_RECOGNIZE_DURATION} seconds")
        start_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")
        extract_audio(
            input_path=audio_path,
            output_path=start_segment_audio_path,
            start_time=f"{timedelta(seconds=0)}",
            duration=env.AUDIO_CHUNK_RECOGNIZE_DURATION
        )

        middle_start = random.randint(
            duration // env.AUDIO_CHUNK_RECOGNIZE_DURATION,
            duration // 3 - env.AUDIO_CHUNK_RECOGNIZE_DURATION
        )
        middle_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")

        extract_audio(
            input_path=audio_path,
            output_path=middle_segment_audio_path,
            start_time=f"{timedelta(seconds=middle_start)}",
            duration=env.AUDIO_CHUNK_RECOGNIZE_DURATION
        )

        end_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")
        extract_audio(
            input_path=audio_path,
            output_path=end_segment_audio_path,
            start_time=f"{timedelta(seconds=duration - env.AUDIO_CHUNK_RECOGNIZE_DURATION * 2)}"
        )

        return start_segment_audio_path, middle_segment_audio_path, end_segment_audio_path

    @staticmethod
    def speech_to_text(audio_path: str, delta: int) -> tuple[str, list[dict]]:
        """
        Converts speech in an audio file to text.

        This method currently supports local speech-to-text conversion using a Whisper model.
        It processes the audio file and returns the recognized text along with timing information.

        Args:
            audio_path (str): The path to the audio file.
            delta (int): A time offset to apply to the start times of recognized segments.

        Returns:
            tuple[str, list[dict]]: A tuple containing:
                - The detected language of the audio (str)
                - A list of dictionaries, each containing:
                    - 'start_time': Start time of the segment in milliseconds (int)
                    - 'duration': Duration of the segment in milliseconds (int)
                    - 'text': The transcribed text for the segment (str)

        Raises:
            AiError: If the audio path is not found or if the speech-to-text provider is not supported.
        """
        if audio_path is None:
            raise AiError("audio path is not found")
        if env.SPEECH_TO_TEXT_PROVIDER == "local":
            model = AiService.__get_local_whisper_model()
            segments, info = model.transcribe(audio=audio_path, beam_size=8, vad_filter=True)
            result = []
            for segment in segments:
                start = (segment.start + delta) * 1000.0
                duration = (segment.end - segment.start) * 1000.0
                logger.debug(
                    f"segment: start: {timedelta(seconds=int(start / 1000.0))}, duration: {int(duration)}ms, text: {segment.text}")
                result.append({
                    'start_time': int(start),
                    'duration': int(duration),
                    'text': segment.text
                })
            return info.language, result
        elif env.SPEECH_TO_TEXT_PROVIDER == "openai":
            raise AiError("Not yet implemented")
        elif env.SPEECH_TO_TEXT_PROVIDER == "gemini":
            raise AiError("Not yet implemented")

    @staticmethod
    def __chunk_text(text: str, max_tokens: int) -> list[str]:
        encoding = tiktoken.get_encoding("cl100k_base")
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_token = encoding.encode(word)
            if current_length + len(word_token) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word_token)
            else:
                current_chunk.append(word)
                current_length += len(word_token)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def get_texts_embedding(provider: str, text: str) -> tuple[list[str], list[list[float]]]:
        """
        Generates text embeddings using the specified provider.

        This method supports multiple embedding providers and routes the request to the appropriate
        embedding function based on the provider specified.

        Args:
            provider (str): The name of the embedding provider to use.
                            Supported values: "gemini", "openai", "voyageai", "mistral", "local"
            text (str): The text to generate embeddings for.

        Returns:
            tuple[list[str], list[list[float]]]: A tuple containing:
                - A list of text chunks (str)
                - A list of embedding vectors (list[float]) for each chunk

        Raises:
            AiError: If an unknown embedding provider is specified.
        """
        if provider == "gemini":
            return AiService.embedding_document_with_gemini(text)
        elif provider == "openai":
            return AiService.embed_document_with_openai(text)
        elif provider == "voyageai":
            return AiService.embed_document_with_voyageai(text)
        elif provider == "mistral":
            return AiService.embed_document_with_mistral(text)
        elif provider == "local":
            return AiService.embed_document_with_local(text)
        else:
            raise AiError("unknown embedding provider")

    @staticmethod
    def embedding_document_with_gemini(text: str, max_tokens=2000) -> tuple[list[str], list[list[float]]]:

        if env.GEMINI_API_KEY is None or env.GEMINI_API_KEY.strip() == "":
            raise AiError("gemini api key is not set or is empty.")
        try:
            texts = AiService.__chunk_text(text, max_tokens)
            genai.configure(api_key=env.GEMINI_API_KEY)
            return texts, [genai.embed_content(content=text, model=env.GEMINI_EMBEDDING_MODEL)['embedding'] for text in
                           texts]
        except Exception as e:
            logger.debug(f"\nerror in embedding_document_with_gemini: \n{text}", exc_info=True)
            raise e

    @staticmethod
    def embed_document_with_openai(text: str, max_tokens=8000) -> tuple[list[str], list[list[float]]]:
        if env.OPENAI_API_KEY is None or env.OPENAI_API_KEY.strip() == "":
            raise AiError("openai api key is not set or is empty.")
        texts = AiService.__chunk_text(text, max_tokens)
        client = OpenAI(api_key=env.OPENAI_API_KEY)
        return texts, [client.embeddings.create(input=[text], model=env.OPENAI_EMBEDDING_MODEL).data[0].embedding for
                       text in texts]

    @staticmethod
    def embed_document_with_voyageai(text: str, max_tokens=16000) -> tuple[list[str], list[list[float]]]:
        if env.VOYAGEAI_API_KEY is None or env.VOYAGEAI_API_KEY.strip() == "":
            raise AiError("voyageai api key is not set or is empty.")
        texts = AiService.__chunk_text(text, max_tokens)
        client = voyageai.Client(api_key=env.VOYAGEAI_API_KEY)
        return texts, [
            client.embed(texts=[text], model=env.VOYAGEAI_EMBEDDING_MODEL, input_type="document").embeddings[0] for text
            in texts]

    @staticmethod
    def embed_document_with_mistral(text: str, max_tokens=8000) -> tuple[list[str], list[list[float]]]:
        if env.MISTRAL_API_KEY is None or env.MISTRAL_API_KEY.strip() == "":
            raise AiError("mistral api key is not set or is empty.")
        texts = AiService.__chunk_text(text, max_tokens)
        client = Mistral(api_key=env.MISTRAL_API_KEY)
        return texts, [client.embeddings.create(inputs=[text], model=env.MISTRAL_EMBEDDING_MODEL).data[0].embedding for
                       text in texts]

    @staticmethod
    def embed_document_with_local(text: str, max_tokens=16000) -> tuple[list[str], list[list[float]]]:
        texts = AiService.__chunk_text(text, max_tokens)
        encoder = AiService.__get_local_embedding_encoder()
        return texts, [encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True).tolist()[0] for text in
                       texts]

    @staticmethod
    def store_embeddings(table: str, ids: list[str], texts: list[str], embeddings: list[list[float]]):
        """
        Stores text embeddings in a Chroma database.

        This method adds or updates embeddings in a specified collection (table) of the Chroma database.

        Args:
            table (str): The name of the collection to store the embeddings in.
            ids (list[str]): A list of unique identifiers for each embedding.
            texts (list[str]): A list of the original text strings corresponding to each embedding.
            embeddings (list[list[float]]): A list of embedding vectors.

        Note:
            The lengths of ids, texts, and embeddings lists must be the same.
        """
        collection = chromadb_client.get_or_create_collection(table)
        collection.add(ids=ids, embeddings=embeddings, documents=texts)

    @staticmethod
    def query_embeddings(table: str, queries: list[list[list[float]]], fetch_size: int = 3, thresholds: list[float] = None) -> tuple[int, list[str]]:
        """
        Queries the Chroma database for similar embeddings.

        This method searches for embeddings in the specified collection (table) that are similar to the provided query embeddings.
        It supports multiple thresholds for filtering results and returns unique documents across all queries.

        Args:
            table (str): The name of the collection to query.
            queries (list[list[list[float]]]): A list of query embedding vectors.
            fetch_size (int, optional): The maximum number of results to return. Defaults to 3.
            thresholds (list[float], optional): A list of similarity thresholds to apply.
                                                If None, uses a default threshold. Defaults to None.

        Returns:
            tuple[int, list[str]]: A tuple containing:
                - The number of unique documents found (int)
                - A list of the found documents (list[str])

        Note:
            The method ensures that duplicate documents are not returned, even if they match multiple queries or thresholds.
        """
        if thresholds is None:
            thresholds = [env.QUERY_SIMILAR_THRESHOLD]
        collection = chromadb_client.get_or_create_collection(table)
        n_result = collection.count()
        top_closest = []
        seen_docs = set()
        for query in queries:
            if not query:
                continue
            results = collection.query(query_embeddings=query, n_results=n_result, include=['documents', 'distances'])

            distances = results['distances']
            documents = results['documents']
            flat_distances = [dist for sublist in distances for dist in sublist]
            flat_documents = [doc for sublist in documents for doc in sublist]

            distance_doc_pairs = list(zip(flat_distances, flat_documents))
            for threshold in thresholds:
                filtered_pairs = [pair for pair in distance_doc_pairs if pair[0] <= threshold]
                sorted_pairs = sorted(filtered_pairs, key=lambda pair: pair[0])
                for data in sorted_pairs:
                    doc_id = sha256(data[1])
                    if doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        top_closest.append(data)

        docs = []
        potential_result = sorted(top_closest, key=lambda pair: pair[0])
        for _, doc in potential_result[:max(1, fetch_size)]:
            docs.append(doc)
        return len(docs), docs

    @staticmethod
    def chat_with_ai(
            provider: str,
            model: str,
            question: str,
            previous_chats: list[dict] = None,
            system_prompt: str = SYSTEM_PROMPT,
            max_tokens: int = 4096,
            temperature: float = 0.6,
            top_p: float = 0.8,
            top_k: int = 16
    ) -> str:
        """
        Interacts with various AI chat models to generate responses.

        This method serves as a unified interface for interacting with different AI providers and models.
        It routes the chat request to the appropriate provider-specific method based on the input.

        Args:
            provider (str): The AI provider to use. Supported values: "gemini", "openai", "claude", "mistral", "ollama"
            model (str): The specific model to use within the chosen provider.
            question (str): The user's input or question to send to the AI.
            previous_chats (list[dict], optional): A list of previous chat messages for context. Defaults to None.
            system_prompt (str, optional): A system message to set the behavior of the AI. Defaults to SYSTEM_PROMPT.
            max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 4096.
            temperature (float, optional): Controls randomness in output. Higher values make output more random. Defaults to 0.6.
            top_p (float, optional): Controls diversity of output. Defaults to 0.8.
            top_k (int, optional): Controls the number of top tokens to consider. Defaults to 16.

        Returns:
            str: The AI-generated response.

        Raises:
            AiError: If an unknown provider is specified.

        Note:
            The specific parameters used (max_tokens, temperature, top_p, top_k) may vary depending on the provider.
        """

        match provider:
            case "gemini":
                return AiService.chat_with_gemini(model, question, previous_chats, system_prompt, max_tokens, temperature, top_p, top_k)
            case "openai":
                return AiService.chat_with_openai(model, question, previous_chats, system_prompt, max_tokens, temperature, top_p)
            case "claude":
                return AiService.chat_with_claude(model, question, previous_chats, system_prompt, max_tokens, temperature, top_p, top_k)
            case "mistral":
                return AiService.chat_with_mistral(model, question, previous_chats, system_prompt, max_tokens, temperature, top_p)
            case "ollama":
                return AiService.chat_with_ollama(model, question, previous_chats, system_prompt, temperature, top_p, top_k)
            case _:
                raise AiError(f"unknown provider: {provider}")

    @staticmethod
    def chat_with_gemini(
            model: str,
            question: str,
            previous_chats: list[dict] = None,
            system_prompt: str = SYSTEM_PROMPT,
            max_tokens: int = 4096,
            temperature: float = 0.6,
            top_p: float = 0.8,
            top_k: int = 16) -> str:
        if previous_chats is None:
            previous_chats = []
        if env.GEMINI_API_KEY is None or env.GEMINI_API_KEY.strip() == "":
            raise AiError("gemini api key is not set or is empty.")
        genai.configure(api_key=env.GEMINI_API_KEY)
        agent = genai.GenerativeModel(
            model_name=model if model is not None or model else "gemini-1.5-flash",
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            ),
            safety_settings={
                'HATE': 'BLOCK_NONE',
                'HARASSMENT': 'BLOCK_NONE',
                'SEXUAL': 'BLOCK_NONE',
                'DANGEROUS': 'BLOCK_NONE'
            }
        )
        chat = agent.start_chat(history=previous_chats) if previous_chats else agent.start_chat()
        response = chat.send_message(question)
        return response.text.removesuffix("\n").strip()

    @staticmethod
    def chat_with_openai(
            model: str,
            question: str,
            previous_chats: list[dict] = None,
            system_prompt: str = SYSTEM_PROMPT,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            top_p: float = 0.8) -> str:

        if previous_chats is None:
            previous_chats = []
        if env.OPENAI_API_KEY is None or env.OPENAI_API_KEY.strip() == "":
            raise AiError("openai api key is not set or is empty.")
        client = OpenAI(api_key=env.OPENAI_API_KEY)
        chat_messages = previous_chats[:]
        chat_messages.insert(0, {
            "role": "system",
            "content": system_prompt
        })
        chat_messages.append({
            "role": "user",
            "content": question
        })
        completion = client.chat.completions.create(
            model=model if model is not None or model else "gpt-4o-mini",
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return completion.choices[0].message.content

    @staticmethod
    def chat_with_claude(
            model: str,
            question: str,
            previous_chats: list[dict] = None,
            system_prompt: str = SYSTEM_PROMPT,
            max_tokens: int = 4096,
            temperature: float = 0.6,
            top_p: float = 0.7,
            top_k: int = 16) -> str:

        if previous_chats is None:
            previous_chats = []
        if env.CLAUDE_API_KEY is None or env.CLAUDE_API_KEY.strip() == "":
            raise AiError("claude api key is not set or is empty.")
        client = anthropic.Anthropic(api_key=env.CLAUDE_API_KEY)
        chat_messages = previous_chats[:]
        chat_messages.append({
            "role": "user", "content": question
        })
        response = client.messages.create(
            model=model if model is not None or model else "claude-3-haiku-20240307",
            system=system_prompt,
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        return response.content[0].text

    @staticmethod
    def chat_with_mistral(
            model: str,
            question: str,
            previous_chats: list[dict] = None,
            system_prompt: str = SYSTEM_PROMPT,
            max_tokens: int = 2048,
            temperature: float = 0.6,
            top_p: float = 0.8) -> str:

        if previous_chats is None:
            previous_chats = []
        if env.MISTRAL_API_KEY is None or env.MISTRAL_API_KEY.strip() == "":
            raise AiError("mistral api key is not set or is empty.")
        client = Mistral(api_key=env.MISTRAL_API_KEY)
        chat_messages = previous_chats[:]
        chat_messages.insert(0, {
            "role": "system", "content": system_prompt
        })
        chat_messages.append({
            "role": "user", "content": question
        })
        response = client.chat.complete(
            model=model if model is not None or model else "mistral-large-latest",
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content

    @staticmethod
    def chat_with_ollama(
            model: str,
            question: str,
            previous_chats: list[dict] = None,
            system_prompt: str = SYSTEM_PROMPT,
            temperature: float = 0.6,
            top_p: float = 0.8,
            top_k: int = 16) -> str:

        if previous_chats is None:
            previous_chats = []
        client = Client(host=env.LOCAL_OLLAMA_HOST)
        chat_messages = previous_chats[:]
        chat_messages.insert(0, {
            "role": "system", "content": system_prompt
        })
        chat_messages.append({
            "role": "user", "content": question
        })
        used_model = model if model else env.LOCAL_OLLAMA_MODEL
        response = client.chat(model=used_model, messages=chat_messages, options={
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        })
        return response['message']['content']
