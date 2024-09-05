import os.path
import random
import tempfile
from collections import Counter
from uuid import uuid4

import google.generativeai as genai
import tiktoken
import voyageai
from audio_extract import extract_audio
from backend import env
from backend.db.models import Chat
from backend.db.specs import chromadb_client
from backend.error.ai_error import AiSegmentError, AiApiKeyError
from backend.utils.logger import log
from backend.utils.prompts import SYSTEM_PROMPT
from faster_whisper import WhisperModel
from future.backports.datetime import timedelta
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class AiService:
    def __init__(self):
        pass

    @staticmethod
    def __get_whisper_model():
        """
        Retrieves a Whisper model instance based on the environment settings.

        Returns:
            WhisperModel: An instance of the Whisper model with the specified device and compute type.
        """
        compute_type = 'int8' if env.LOCAL_WHISPER_DEVICE == 'cpu' else 'fp16'
        return WhisperModel(env.LOCAL_WHISPER_MODEL, device=env.LOCAL_WHISPER_DEVICE, compute_type=compute_type)

    @staticmethod
    def __get_local_embedding_encoder():
        """
        Retrieves a local SentenceTransformer model instance based on the environment settings.

        If the model is not found locally, it is downloaded and saved to the specified path.

        Returns:
            SentenceTransformer: An instance of the SentenceTransformer model with the specified device and model path.
        """
        local_model_path: str = os.path.join(env.APP_DIR, env.LOCAL_EMBEDDING_MODEL)
        if not os.path.exists(local_model_path):
            encoder = SentenceTransformer(model_name_or_path=env.LOCAL_EMBEDDING_MODEL, device=env.LOCAL_EMBEDDING_DEVICE, trust_remote_code=True)
            encoder.save(local_model_path)
            return encoder
        return SentenceTransformer(model_name_or_path=local_model_path, device=env.LOCAL_EMBEDDING_DEVICE, trust_remote_code=True)

    def recognize_audio_language(self, audio_path, duration):
        """
        Recognizes the language of an audio file.

        Note:
            - If audio length is less than 120 seconds, use whole audio to detect. Otherwise, random pick a set of split audios (at the end, middle, start) to detect language.
            - Current faster-whisper does not support any "fast way" detecting languages. Here is my work-around solution.

        Args:
            audio_path (str): The path to the audio file.
            duration (int): The duration of the audio file in seconds.

        Returns:
            str: The language of the audio file, or None if it cannot be determined.
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
        """
        Segments an audio file into three random parts: start, middle, and end.

        Args:
            audio_path (str): The path to the audio file.
            duration (int): The duration of the audio file in seconds.

        Returns:
            tuple: A tuple containing the paths to the start, middle, and end segments of the audio file.

        Raises:
            AiSegmentError: If the duration of the audio file is less than 120 seconds.
        """
        if duration < 120:
            raise AiSegmentError("Duration must be greater than 120 seconds")
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
        """
        Transcribes an audio file into text.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            list: A list of dictionaries containing the start time (milliseconds), duration (milliseconds), and text of each segment in the audio file.
        """
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
        """
        Splits a given text into chunks of words, each chunk containing a maximum number of tokens.

        Args:
            text (str): The text to be chunked.
            max_tokens (int): The maximum number of tokens in each chunk.

        Returns:
            List[str]: A list of chunks, where each chunk is a string of words.
        """
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
        """
        Embeds a document using the Gemini API.

        Args:
            text (str): The text to be embedded.
            max_tokens (int, optional): The maximum number of tokens in the embedded text. Defaults to 2000.

        Returns:
            tuple: A tuple containing the chunked text and the embeddings.

        Raises:
            AiApiKeyError: If the Gemini API key is not set or is empty.
        """
        if env.GEMINI_API_KEY is None or env.GEMINI_API_KEY.strip() == "":
            raise AiApiKeyError("Gemini API key is not set or is empty.")
        try:
            texts = AiService.__chunk_text(text, max_tokens)
            genai.configure(api_key=env.GEMINI_API_KEY)
            return texts, [genai.embed_content(model=env.GEMINI_EMBEDDING_MODEL, content=text)['embedding'] for text in texts]
        except Exception as e:
            log.debug(f"\nError in embedding_document_with_gemini: \n{text}", exc_info=True)
            raise e

    @staticmethod
    def embedding_document_with_openai(text: str, max_tokens=8000):
        """
        Embeds a document using the OpenAI API.

        Args:
            text (str): The text to be embedded.
            max_tokens (int, optional): The maximum number of tokens in each chunk. Defaults to 8000.

        Returns:
            tuple: A tuple containing the chunked text and the embeddings.

        Raises:
            AiApiKeyError: If the OpenAI API key is not set or is empty.
        """
        if env.OPENAI_API_KEY is None or env.OPENAI_API_KEY.strip() == "":
            raise AiApiKeyError("OpenAI API key is not set or is empty.")
        texts = AiService.__chunk_text(text, max_tokens)
        client = OpenAI(api_key=env.OPENAI_API_KEY)
        return texts, [client.embeddings.create(model=env.OPENAI_EMBEDDING_MODEL, input=[text]).data[0].embedding for text in texts]

    @staticmethod
    def embedding_document_with_voyageai(text: str, max_tokens=16000):
        """
        Embeds a document using the VoyageAI API.

        Args:
            text (str): The text to be embedded.
            max_tokens (int, optional): The maximum number of tokens in each chunk. Defaults to 16000.

        Returns:
            tuple: A tuple containing the chunked text and the embeddings.

        Raises:
            AiApiKeyError: If the VoyageAI API key is not set or is empty.
        """
        if env.VOYAGEAI_API_KEY is None or env.VOYAGEAI_API_KEY.strip() == "":
            raise AiApiKeyError("VoyageAI API key is not set or is empty.")
        texts = AiService.__chunk_text(text, max_tokens)
        client = voyageai.Client(api_key=env.VOYAGEAI_API_KEY)
        return texts, [client.embed(texts=[text], model=env.VOYAGEAI_EMBEDDING_MODEL, input_type="document").embeddings[0] for text in texts]

    @staticmethod
    def embedding_document_with_local(text: str, max_tokens=500):
        """
        Embeds a document using a local embedding model.

        Args:
            text (str): The text to be embedded.
            max_tokens (int, optional): The maximum number of tokens in each chunk. Defaults to 500, depends on local model that used.

        Returns:
            tuple: A tuple containing the chunked text and the embeddings.
        """
        texts = AiService.__chunk_text(text, max_tokens)
        encoder = AiService.__get_local_embedding_encoder()
        return texts, [encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True).tolist()[0] for text in texts]

    @staticmethod
    def store_embeddings(table: str, ids: list[str], texts: list[str], embeddings: list[list[float]]):
        """
        Store embeddings in a ChromaDB collection.

        Args:
            table (str): The name of the collection to store the embeddings in.
            ids (list[str]): The list of IDs associated with the embeddings.
            texts (list[str]): The list of texts associated with the embeddings.
            embeddings (list[list[float]]): The list of embeddings to store.

        Returns:
            None
        """
        collection = chromadb_client.get_or_create_collection(table)
        collection.add(ids=ids, embeddings=embeddings, documents=texts)

    @staticmethod
    def query_embeddings(table: str, query: list[list[float]], fetch_size: int = 10, thresholds: list[float] = None):
        """
        Query the embeddings in a ChromaDB collection based on the given query embeddings.

        Try to add all valid documents that has distance less than threshold, then remove duplication doucment.

        Args:
            table (str): The name of the collection to query the embeddings from.
            query (list[list[float]]): The list of query embeddings.
            fetch_size (int, optional): The number of results to fetch. Defaults to 10.
            thresholds (list[float], optional): The list of thresholds to filter the results. Defaults to [0.3, 0.6].

        Returns:
            tuple: A tuple containing the count of unique documents and a string of the unique documents joined by newline characters.
        """
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
            top_k: int = 32):
        """
        Initiates a conversation with the Gemini AI model.

        Args:
            model (str): The name of the Gemini model to use.
            prompt (str): The initial message to send to the model.
            system_prompt (str, optional): The system prompt to provide to the model. Defaults to SYSTEM_PROMPT.
            previous_chats (list[Chat], optional): A list of previous chat messages. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 4096.
            temperature (float, optional): The temperature to use for generation. Defaults to 0.6.
            top_p (float, optional): The top p value to use for generation. Defaults to 0.6.
            top_k (int, optional): The top k value to use for generation. Defaults to 32.

        Returns:
            str: The response
        """
        if previous_chats is None:
            previous_chats = []
        if env.GEMINI_API_KEY is None or env.GEMINI_API_KEY.strip() == "":
            raise AiApiKeyError("Gemini API key is not set or is empty.")
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
            top_p: float = 0.8):
        """
        Initiates a conversation with OpenAI's chat completion API.

        Args:
            model (str): The model to use for the conversation.
            prompt (str): The initial message to send to the model.
            system_prompt (str, optional): The system prompt to use for the conversation. Defaults to SYSTEM_PROMPT.
            previous_chats (list[Chat], optional): A list of previous chats to include in the conversation history
        """
        if previous_chats is None:
            previous_chats = []
        if env.OPENAI_API_KEY is None or env.OPENAI_API_KEY.strip() == "":
            raise AiApiKeyError("OpenAI API key is not set or is empty.")
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
        """
        Initiates a conversation with Claude's API.

        Args:
            model (str): The model to use for the conversation.
            prompt (str): The initial message to send to the model.
            system_prompt (str, optional): The system prompt to use for the conversation. Defaults to SYSTEM_PROMPT.
            previous_chats (list[Chat], optional): A list of previous chats to include in the conversation history. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 4096.
            temperature (float, optional): The temperature to use for the conversation. Defaults to 0.7.
            top_p (float, optional): The top p value to use for the conversation. Defaults to 0.7.
            top_k (int, optional): The top k value to use for the conversation. Defaults to 16.

        Returns:
            str: The response from Claude's API.
        """
        if previous_chats is None:
            previous_chats = []
        if env.CLAUDE_API_KEY is None or env.CLAUDE_API_KEY.strip() == "":
            raise AiApiKeyError("Claude API key is not set or is empty.")
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
        """
        Initiates a conversation with OLLAMA's chat completion API.

        Args:
            model (str): The model to use for the conversation.
            prompt (str): The initial message to send to the model.
            system_prompt (str, optional): The system prompt to use for the conversation. Defaults to SYSTEM_PROMPT.
            previous_chats (list[Chat], optional): A list of previous chats to include in the conversation history. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 2048.
            temperature (float, optional): The temperature to use for the response generation. Defaults to 0.7.
            top_p (float, optional): The top p value to use for the response generation. Defaults to 1.0.

        Returns:
            None

        Raises:
            NotImplementedError: OLLAMA is not implemented.
        """
        if previous_chats is None:
            previous_chats = []
        raise NotImplementedError("OLLAMA is not implemented")
