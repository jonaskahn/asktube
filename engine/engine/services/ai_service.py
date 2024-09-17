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

from engine.database.models import Chat
from engine.database.specs import chromadb_client
from engine.supports import env
from engine.supports.env import MISTRAL_API_KEY
from engine.supports.errors import AiError
from engine.supports.prompts import SYSTEM_PROMPT

has_cuda = torch.cuda.is_available()
has_mps = torch.backends.mps.is_available()


class AiService:
    def __init__(self):
        pass

    @staticmethod
    def __get_local_whisper_model():
        """
        Retrieves a WhisperModel instance based on the environment settings.

        The model's compute type is determined by the availability of CUDA,
        defaulting to 'fp16' if CUDA is available and 'int8' if CPU.

        Returns:
            WhisperModel: An instance of the WhisperModel with the specified device and model path.
        """

        compute_type = 'fp16' if has_cuda else 'int8'
        return WhisperModel(
            env.LOCAL_WHISPER_MODEL,
            device=env.LOCAL_WHISPER_DEVICE,
            compute_type=compute_type,
            download_root=env.APP_DIR
        )

    @staticmethod
    def __get_local_embedding_encoder():
        """
        Retrieves a SentenceTransformer instance based on the environment settings.

        The device type is determined by the availability of MPS or CUDA, defaulting to 'cpu' if neither is available.
        If the local model path does not exist, it downloads the model and saves it to the local path.

        Returns:
            SentenceTransformer: An instance of the SentenceTransformer with the specified device and model path.
        """

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
    def recognize_audio_language(audio_path: str, duration: int):
        """
        Recognizes the language of an audio file.

        Args:
            audio_path (str): The path to the audio file.
            duration (int): The duration of the audio file in seconds.

        Returns:
            str: The recognized language of the audio file, or None if the language could not be determined.
        """

        logger.debug("start to recognize audio language")
        model = AiService.__get_local_whisper_model()
        if duration <= 120:
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
    def __split_segment_audio(audio_path: str, duration: int):
        """
        Segments an audio file into three parts: start, middle, and end.

        Args:
            audio_path (str): The path to the audio file.
            duration (int): The duration of the audio file in seconds.

        Returns:
            tuple: A tuple containing the paths to the start, middle, and end segments of the audio file.
        """

        if duration < 120:
            raise AiError("duration must be greater than 120 seconds")
        start_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")
        extract_audio(
            input_path=audio_path,
            output_path=start_segment_audio_path,
            start_time=f"{timedelta(seconds=0)}",
            duration=env.AUDIO_CHUNK_DETECT_DURATION
        )

        middle_start = random.randint(
            duration // env.AUDIO_CHUNK_DETECT_DURATION,
            duration // 3 - env.AUDIO_CHUNK_DETECT_DURATION
        )
        middle_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")

        extract_audio(
            input_path=audio_path,
            output_path=middle_segment_audio_path,
            start_time=f"{timedelta(seconds=middle_start)}",
            duration=env.AUDIO_CHUNK_DETECT_DURATION
        )

        end_segment_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.mp3")
        extract_audio(
            input_path=audio_path,
            output_path=end_segment_audio_path,
            start_time=f"{timedelta(seconds=duration - env.AUDIO_CHUNK_DETECT_DURATION * 2)}"
        )

        return start_segment_audio_path, middle_segment_audio_path, end_segment_audio_path

    @staticmethod
    def speech_to_text(audio_path: str, delta: int):
        """
        Transcribes the given audio file to text using the specified speech-to-text provider.

        Args:
            audio_path (str): The path to the audio file.
            delta (int): The delta value to adjust the start time of each segment.

        Returns:
            tuple: A tuple containing the language of the transcription and a list of segments.
                   Each segment is a dictionary with 'start_time', 'duration', and 'text' keys.
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
        """
        Chunks the given text into a list of substrings, each containing a maximum number of tokens.

        Args:
            text (str): The text to be chunked.
            max_tokens (int): The maximum number of tokens allowed in each chunk.

        Returns:
            list[str]: A list of chunked substrings.
        """

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
    def embedding_document_with_gemini(text: str, max_tokens=2000) -> tuple[list[str], list[list[float]]]:
        """
        Embeds a document using the Gemini API.

        Args:
            text (str): The text to be embedded.
            max_tokens (int, optional): The maximum number of tokens in each chunk. Defaults to 2000.

        Returns:
            tuple: A tuple containing the chunked text and the embeddings.

        Raises:
            AiError: If the Gemini API key is not set or is empty.
        """

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
        """
        Embeds a document using the OpenAI API.

        Args:
            text (str): The text to be embedded.
            max_tokens (int, optional): The maximum number of tokens in each chunk. Defaults to 8000.

        Returns:
            tuple: A tuple containing the chunked text and the embeddings.

        Raises:
            AiError: If the OpenAI API key is not set or is empty.
        """

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
        """
        Embeds a document using the Mistral API.

        Args:
            text (str): The text to be embedded.
            max_tokens (int, optional): The maximum number of tokens in each chunk. Defaults to 8000.

        Returns:
            tuple: A tuple containing the chunked text and the embeddings.

        Raises:
            AiError: If the Mistral API key is not set or is empty.
        """

        if env.MISTRAL_API_KEY is None or env.MISTRAL_API_KEY.strip() == "":
            raise AiError("mistral api key is not set or is empty.")
        texts = AiService.__chunk_text(text, max_tokens)
        client = Mistral(api_key=MISTRAL_API_KEY)
        return texts, [client.embeddings.create(inputs=[text], model=env.MISTRAL_EMBEDDING_MODEL).data[0].embedding for
                       text in texts]

    @staticmethod
    def embed_document_with_local(text: str, max_tokens=16000) -> tuple[list[str], list[list[float]]]:
        """
        Embeds a document using a local embedding encoder.

        Args:
            text (str): The text to be embedded.
            max_tokens (int, optional): The maximum number of tokens in each chunk. Defaults to 16000.

        Returns:
            tuple: A tuple containing the chunked text and the embeddings.
        """

        texts = AiService.__chunk_text(text, max_tokens)
        encoder = AiService.__get_local_embedding_encoder()
        return texts, [encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True).tolist()[0] for text in
                       texts]

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
        Queries embeddings in a ChromaDB collection.

        Args:
            table (str): The name of the collection to query the embeddings from.
            query (list[list[float]]): The list of query embeddings.
            fetch_size (int, optional): The number of results to fetch. Defaults to 10.
            thresholds (list[float], optional): The list of thresholds to filter the results by. Defaults to [env.QUERY_SIMILAR_THRESHOLD].

        Returns:
            tuple: A tuple containing the number of unique documents and the unique documents themselves, joined by newline characters.
        """

        if thresholds is None:
            thresholds = [env.QUERY_SIMILAR_THRESHOLD]
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
            top_k: int = 32) -> str:
        """
        Initiates a conversation with the Gemini AI model.

        Args:
            model (str): The name of the Gemini model to use.
            prompt (str): The initial message to send to the model.
            system_prompt (str, optional): The system prompt to use. Defaults to SYSTEM_PROMPT.
            previous_chats (list[Chat], optional): A list of previous chat messages. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 4096.
            temperature (float, optional): The temperature to use for generation. Defaults to 0.6.
            top_p (float, optional): The top p value to use for generation. Defaults to 0.6.
            top_k (int, optional): The top k value to use for generation. Defaults to 32.

        Returns:
            str: The response from the Gemini model.
        """

        if previous_chats is None:
            previous_chats = []
        if env.GEMINI_API_KEY is None or env.GEMINI_API_KEY.strip() == "":
            raise AiError("gemini api key is not set or is empty.")
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
            ),
            safety_settings={
                'HATE': 'BLOCK_NONE',
                'HARASSMENT': 'BLOCK_NONE',
                'SEXUAL': 'BLOCK_NONE',
                'DANGEROUS': 'BLOCK_NONE'
            }
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
            top_p: float = 0.8) -> str:
        """
        Sends a prompt to OpenAI's chat completion API and returns the response.

        Args:
            model (str): The model to use for the chat completion. Defaults to "gpt-4o-mini" if not provided.
            prompt (str): The prompt to send to the chat completion API.
            system_prompt (str, optional): The system prompt to use for the chat completion. Defaults to SYSTEM_PROMPT.
            previous_chats (list[Chat], optional): A list of previous chats to include in the chat history. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 4096.
            temperature (float, optional): The temperature to use for the chat completion. Defaults to 0.7.
            top_p (float, optional): The top_p value to use for the chat completion. Defaults to 0.8.

        Returns:
            str: The response from the chat completion API.
        """

        if previous_chats is None:
            previous_chats = []
        if env.OPENAI_API_KEY is None or env.OPENAI_API_KEY.strip() == "":
            raise AiError("openai api key is not set or is empty.")
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
            top_k: int = 16) -> str:
        """
        Initiates a conversation with Claude AI model.

        Args:
            model (str): The Claude model to use for the conversation.
            prompt (str): The user's prompt for the conversation.
            system_prompt (str, optional): The system prompt for the conversation. Defaults to SYSTEM_PROMPT.
            previous_chats (list[Chat], optional): A list of previous chats to include in the conversation history. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 4096.
            temperature (float, optional): The temperature to use for the response generation. Defaults to 0.7.
            top_p (float, optional): The top p value to use for the response generation. Defaults to 0.7.
            top_k (int, optional): The top k value to use for the response generation. Defaults to 16.

        Returns:
            str: The response from Claude AI model.
        """

        if previous_chats is None:
            previous_chats = []
        if env.CLAUDE_API_KEY is None or env.CLAUDE_API_KEY.strip() == "":
            raise AiError("claude api key is not set or is empty.")
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
    def chat_with_mistral(
            model: str,
            prompt: str,
            system_prompt: str = SYSTEM_PROMPT,
            previous_chats: list[Chat] = None,
            max_tokens: int = 2048,
            temperature: float = 0.7,
            top_p: float = 1.0) -> str:
        """
        Sends a chat request to Mistral AI model and returns the response.

        Args:
            model (str): The Mistral AI model to use for the chat.
            prompt (str): The user's prompt for the chat.
            system_prompt (str): The system's prompt for the chat. Defaults to SYSTEM_PROMPT.
            previous_chats (list[Chat]): A list of previous chats to include in the conversation. Defaults to None.
            max_tokens (int): The maximum number of tokens to generate in the response. Defaults to 2048.
            temperature (float): The temperature to use for the response generation. Defaults to 0.7.
            top_p (float): The top-p value to use for the response generation. Defaults to 1.0.

        Returns:
            str: The response from the Mistral AI model.
        """

        if previous_chats is None:
            previous_chats = []
        if env.MISTRAL_API_KEY is None or env.MISTRAL_API_KEY.strip() == "":
            raise AiError("mistral api key is not set or is empty.")
        client = Mistral(api_key=env.MISTRAL_API_KEY)
        messages = AiService.__build_mistral_chat_history(system_prompt=system_prompt, chats=previous_chats)
        messages.append({
            "role": "user", "content": prompt
        })
        response = client.chat.complete(
            model=model if model is not None or model else "mistral-large-latest",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content

    @staticmethod
    def __build_mistral_chat_history(system_prompt: str, chats: list[Chat]):
        return AiService.__build_openai_chat_history(system_prompt=system_prompt, chats=chats)

    @staticmethod
    def chat_with_ollama(
            model: str,
            prompt: str,
            system_prompt: str = SYSTEM_PROMPT,
            previous_chats: list[Chat] = None,
            temperature: float = 0.7,
            top_p: float = 1.0,
            top_k: int = 16) -> str:
        """
        Initiates a conversation with the Ollama AI model.

        Args:
            model (str): The model to use for the conversation.
            prompt (str): The initial prompt to send to the model.
            system_prompt (str, optional): The system prompt to use for the conversation. Defaults to SYSTEM_PROMPT.
            previous_chats (list[Chat], optional): A list of previous chats to include in the conversation. Defaults to None.
            temperature (float, optional): The temperature to use for the response generation. Defaults to 0.7.
            top_p (float, optional): The top-p value to use for the response generation. Defaults to 1.0.
            top_k (int, optional): The top-k value to use for the response generation. Defaults to 16.

        Returns:
            str: The response from the Ollama AI model.
        """

        if previous_chats is None:
            previous_chats = []
        client = Client(host=env.LOCAL_OLLAMA_HOST)
        messages = AiService.__build_ollama_chat_history(system_prompt=system_prompt, chats=previous_chats)
        messages.append({
            "role": "user", "content": prompt
        })
        response = client.chat(model=env.LOCAL_OLLAMA_MODEL, messages=messages, options={
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        })
        return response['message']['content']

    @staticmethod
    def __build_ollama_chat_history(system_prompt: str, chats: list[Chat]):
        return AiService.__build_openai_chat_history(system_prompt=system_prompt, chats=chats)

    @staticmethod
    def delete_collection(table: str):
        chromadb_client.delete_collection(table)
