import iso639
from playhouse.shortcuts import model_to_dict
from retry import retry
from sanic.log import logger

from engine.database.models import Video, Chat
from engine.services.ai_service import AiService
from engine.services.video_service import VideoService
from engine.supports import env
from engine.supports.errors import ChatError
from engine.supports.prompts import ASKING_PROMPT_WITH_RAG, ASKING_PROMPT_WITHOUT_RAG, MULTI_QUERY_PROMPT, SYSTEM_PROMPT


class ChatService:
    """
    A service class that handles chat interactions with AI models for video content.
    This class provides methods for asking questions about videos, managing chat histories,
    and interacting with various AI providers using different retrieval techniques.
    """

    def __init__(self):
        pass

    @staticmethod
    async def ask(question: str, vid: int, provider: str, model: str = None) -> dict[object]:
        """
        Processes a user's question about a specific video using the specified AI provider and model.

        This method determines whether to use RAG (Retrieval-Augmented Generation) based on the video's
        transcript length. It then routes the question to the appropriate processing method.

        Logic:
        1. Retrieves the video and recent chat history.
        2. Checks if the video's transcript is short enough for direct processing.
        3. Calls either __ask_without_rag or __ask_with_rag based on the transcript length.

        Args:
            question (str): The user's question about the video.
            vid (int): The ID of the video being queried.
            provider (str): The AI provider to use (e.g., "gemini", "openai").
            model (str, optional): The specific model to use. Defaults to None.

        Returns:
            dict[object]: A dictionary representation of the Chat object containing the question and answer.

        Raises:
            May raise exceptions related to video retrieval or AI processing.
        """
        video: Video = VideoService.get_analyzed_video(vid)
        chats: list[Chat] = list(Chat.select().where(Chat.video == video).limit(20).order_by(Chat.id.asc()))
        if video.transcript_tokens <= env.TOKEN_CONTEXT_THRESHOLD:
            return await ChatService.__ask_without_rag(question, video, chats, provider, model)
        else:
            return await ChatService.__ask_with_rag(question, video, chats, provider, model)

    @staticmethod
    async def __ask_without_rag(question: str, video: Video, chats: list[Chat], provider: str, model: str = None) -> dict[object]:
        """
        Processes a question about a video without using RAG, suitable for shorter transcripts.

        This method constructs a context from the video details and full transcript, then sends
        the question to the appropriate AI provider for processing.

        Logic:
        1. Constructs the context using the ASKING_PROMPT_WITHOUT_RAG template.
        2. Routes the question to the specific AI provider's method.
        3. Creates and saves a new Chat object with the result.

        Args:
            question (str): The user's question.
            video (Video): The Video object being queried.
            chats (list[Chat]): List of previous chat messages.
            provider (str): The AI provider to use.
            model (str, optional): The specific model to use. Defaults to None.

        Returns:
            dict[object]: A dictionary representation of the new Chat object.

        Raises:
            ChatError: If an unsupported provider is specified.
        """
        context = ASKING_PROMPT_WITHOUT_RAG.format(**{
            "title": video.title,
            "url": video.url,
            "description": video.description,
            "context": video.transcript
        })
        match provider:
            case "gemini":
                result = await ChatService.__ask_gemini_without_rag(model=model, question=question, context=context, chats=chats)
            case "openai":
                result = await ChatService.__ask_openai_without_rag(model=model, question=question, context=context, chats=chats)
            case "claude":
                result = await ChatService.__ask_claude_without_rag(model=model, question=question, context=context, chats=chats)
            case "mistral":
                result = await ChatService.__ask_mistral_without_rag(model=model, question=question, context=context, chats=chats)
            case "ollama":
                result = await ChatService.__ask_ollama_without_rag(model=model, question=question, context=context, chats=chats)
            case _:
                raise ChatError("provider is not supported")

        chat = Chat.create(
            video=video,
            question=question,
            refined_question="Not implemented yet",
            answer=result,
            context="No context without RAG",
            prompt=context,
            provider=provider
        )
        chat.save()
        return model_to_dict(chat)

    @staticmethod
    async def __ask_gemini_without_rag(model, question, context, chats):
        previous_chats = ChatService.__build_gemini_without_rag_chat_histories(context=context, chats=chats)
        return AiService.chat_with_gemini(model=model, question=question, previous_chats=previous_chats)

    @staticmethod
    def __build_gemini_without_rag_chat_histories(context: str, chats: list[Chat]) -> list[dict]:
        chat_histories = [
            {
                "role": "user",
                "parts": context,
            },
            {
                "role": "model",
                "parts": "I read carefully the video information and what you provided, let's go QA",
            }
        ]
        if chats is not None:
            for chat in chats:
                chat_histories.extend((
                    {"role": "user", "parts": chat.question},
                    {"role": "model", "parts": chat.answer},
                ))
        return chat_histories

    @staticmethod
    async def __ask_openai_without_rag(model, question, context, chats) -> str:
        previous_chats = ChatService.__build_openai_without_rag_chat_histories(context=context, chats=chats)
        return AiService.chat_with_gemini(model=model, question=question, previous_chats=previous_chats)

    @staticmethod
    def __build_openai_without_rag_chat_histories(context, chats) -> list[dict]:
        chat_histories = [
            {
                "role": "user",
                "content": context,
            },
            {
                "role": "assistant",
                "content": "I read carefully the video information and what you provided, let's go QA",
            }
        ]
        if chats is not None:
            for chat in chats:
                chat_histories.extend(
                    (
                        {"role": "user", "content": chat.question},
                        {"role": "assistant", "content": chat.answer},
                    )
                )
        return chat_histories

    @staticmethod
    async def __ask_claude_without_rag(model, question, context, chats):
        previous_chats = ChatService.__build_openai_without_rag_chat_histories(context=context, chats=chats)
        return AiService.chat_with_claude(model=model, question=question, previous_chats=previous_chats)

    @staticmethod
    async def __ask_mistral_without_rag(model, question, context, chats):
        previous_chats = ChatService.__build_openai_without_rag_chat_histories(context=context, chats=chats)
        return AiService.chat_with_mistral(model=model, question=question, previous_chats=previous_chats)

    @staticmethod
    async def __ask_ollama_without_rag(model, question, context, chats):
        previous_chats = ChatService.__build_openai_without_rag_chat_histories(context=context, chats=chats)
        return AiService.chat_with_ollama(model=model, question=question, previous_chats=previous_chats)

    @staticmethod
    async def __ask_with_rag(question: str, video: Video, chats: list[Chat], provider: str, model: str = None) -> dict[object]:
        """
        Processes a question about a video using RAG, suitable for longer transcripts.

        This method retrieves relevant document snippets based on the question, constructs a context,
        and then sends the question to the appropriate AI provider for processing.

        Logic:
        1. Retrieves relevant document snippets using __get_relevant_doc.
        2. Constructs the context using the ASKING_PROMPT_WITH_RAG template or a default prompt.
        3. Routes the question to the specific AI provider's method.
        4. Creates and saves a new Chat object with the result.

        Args:
            question (str): The user's question.
            video (Video): The Video object being queried.
            chats (list[Chat]): List of previous chat messages.
            provider (str): The AI provider to use.
            model (str, optional): The specific model to use. Defaults to None.

        Returns:
            dict[object]: A dictionary representation of the new Chat object.

        Raises:
            ChatError: If an unsupported provider is specified.
        """
        previous_chats = list(Chat.select().where(Chat.video == video).limit(5).order_by(Chat.id.asc()))
        previous_questions = "\n".join([chat.question for chat in previous_chats])
        context_document = ChatService.__get_relevant_doc(provider=provider, model=model, video=video, question=question, previous_questions=previous_questions)
        logger.debug(f"Relevant docs: {context_document}")
        prompt_context = ASKING_PROMPT_WITH_RAG.format(**{"context": context_document}) if context_document else None

        if not prompt_context and env.RAG_AUTO_SWITCH in ["on", "yes", "enabled"]:
            logger.debug("RAG is required, but none relevant information found, auto switch")
            return await ChatService.__ask_without_rag(question=question, video=video, chats=chats, provider=provider, model=model)

        awareness_context = prompt_context if prompt_context else "No video information related, just answer me in your ability"
        match provider:
            case "gemini":
                result = await ChatService.__ask_gemini_with_rag(model=model, question=question, context=awareness_context, chats=chats)
            case "openai":
                result = await ChatService.__ask_openai_with_rag(model=model, question=question, context=awareness_context, chats=chats)
            case "claude":
                result = await ChatService.__ask_claude_with_rag(model=model, question=question, context=awareness_context, chats=chats)
            case "mistral":
                result = await ChatService.__ask_mistral_with_rag(model=model, question=question, context=awareness_context, chats=chats)
            case "ollama":
                result = await ChatService.__ask_ollama_with_rag(model=model, question=question, context=awareness_context, chats=chats)
            case _:
                raise ChatError("unknown chat provider")

        chat = Chat.create(
            video=video,
            question=question,
            refined_question="Not Implemented Yet",
            answer=result,
            relevant_docs=context_document if context_document else "No context doc found",
            prompt=prompt_context if prompt_context else "No prompt found",
            provider=provider
        )
        chat.save()
        return model_to_dict(chat)

    @staticmethod
    def __get_relevant_doc(provider: str, model: str, video: Video, question: str, previous_questions: str) -> str | None:
        """
        Retrieves relevant document snippets based on the question and video content.

        This method uses the configured RAG query implementation to find relevant parts of the video transcript.
        Currently, it only supports the "multiquery" implementation.

        Logic:
        1. Checks the configured RAG query implementation.
        2. Calls the appropriate method based on the implementation (currently only __get_relevant_doc_by_multiquery).

        Args:
            provider (str): The AI provider to use.
            model (str): The specific model to use.
            video (Video): The Video object being queried.
            question (str): The user's question.

        Returns:
            str | None: A string containing relevant document snippets, or None if no relevant snippets are found.

        Raises:
            ChatError: If an unsupported RAG query implementation is configured.
        """
        implementation = env.RAG_QUERY_IMPLEMENTATION
        match implementation:
            case "multiquery":
                return ChatService.__get_relevant_doc_by_multiquery(provider, model, video, question, previous_questions)
            case _:
                raise ChatError(f"not support {implementation} yet")

    @staticmethod
    def __get_relevant_doc_by_multiquery(provider: str, model: str, video: Video, question: str, previous_questions: str) -> str | None:
        """
        Retrieves relevant document snippets using the multi-query approach.

        This method generates multiple queries based on the original question and uses them to find relevant
        parts of the video transcript.

        Logic:
        1. Constructs a multi-query prompt using the question and video details.
        2. Uses the AI to generate multiple related questions.
        3. If the first generated question is identical to the original, returns None.
        4. Otherwise, calls __query_document_by_multi_query to find relevant snippets.

        Args:
            provider (str): The AI provider to use.
            model (str): The specific model to use.
            video (Video): The Video object being queried.
            question (str): The user's original question.

        Returns:
            str | None: A string containing relevant document snippets, or None if no relevant snippets are found.
        """
        multi_query_prompt = MULTI_QUERY_PROMPT.format(**{
            "title": video.title,
            "description": video.description,
            "previous_questions": previous_questions if previous_questions else "NO PREVIOUS QUESTIONS",
            "question": question,
            "language": iso639.Language.from_part1(video.language).name
        })
        logger.debug(f"Multiquery generated:```\n{multi_query_prompt}\n```")
        questions = AiService.chat_with_ai(provider=provider, model=model, question=multi_query_prompt, system_prompt=None).split("\n")
        if len(questions) == 1 and questions[0].lower() == "0":
            return None

        questions.append(question)
        return ChatService.__query_document_by_multi_query(questions, video)

    @staticmethod
    @retry(tries=5)
    def __query_document_by_multi_query(questions, video):
        """
        Queries the video transcript using multiple generated questions.

        This method converts each question into an embedding and uses these embeddings to find
        relevant parts of the video transcript.

        Logic:
        1. Generates embeddings for each non-empty question.
        2. Queries the video's embedding database using these embeddings.
        3. Combines the retrieved documents into a single string.

        Args:
            questions (list[str]): List of generated questions.
            video (Video): The Video object being queried.

        Returns:
            str | None: A string containing relevant document snippets, or None if no relevant snippets are found.

        Note:
            This method is decorated with @retry to attempt the operation up to 5 times in case of failure.
        """
        embedding_questions = []
        for relevant_question in questions:
            if relevant_question and relevant_question.strip():
                logger.debug(f"question: {relevant_question}")
                _, embedding_question = AiService.get_texts_embedding(video.embedding_provider, relevant_question)
                embedding_questions.append(embedding_question)
        _, documents = AiService.query_embeddings(
            table=f"video_{video.id}",
            queries=embedding_questions
        )
        return "\n".join(documents) if documents else None

    @staticmethod
    async def __ask_gemini_with_rag(model: str, question: str, context: str, chats: list[Chat]) -> str:
        previous_chats = ChatService.__build_gemini_rag_chat_histories(question=question, chats=chats)
        return AiService.chat_with_gemini(model=model, question=context, previous_chats=previous_chats, system_prompt=SYSTEM_PROMPT)

    @staticmethod
    def __build_gemini_rag_chat_histories(question: str, chats: list[Chat]) -> list[dict]:
        chat_histories = []
        if chats is not None:
            for chat in chats:
                chat_histories.extend((
                    {"role": "user", "parts": chat.question},
                    {"role": "model", "parts": chat.answer},
                ))
        chat_histories.extend((
            {"role": "user", "parts": question},
            {"role": "model", "parts": "Could you provide me related information for your request"}

        ))
        return chat_histories

    @staticmethod
    async def __ask_openai_with_rag(model: str, question: str, context: str, chats: list[Chat]) -> str:
        previous_chats = ChatService.__build_openai_rag_chat_histories(question=question, chats=chats)
        return AiService.chat_with_openai(model=model, question=context, previous_chats=previous_chats, system_prompt=SYSTEM_PROMPT)

    @staticmethod
    def __build_openai_rag_chat_histories(question: str, chats: list[Chat]) -> list[dict]:
        chat_histories = []
        if chats is not None:
            for chat in chats:
                chat_histories.extend(
                    (
                        {"role": "user", "content": chat.question},
                        {"role": "assistant", "content": chat.answer},
                    )
                )
        chat_histories.extend((
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Could you provide me related information for your request"}

        ))
        return chat_histories

    @staticmethod
    async def __ask_claude_with_rag(model: str, question: str, context: str, chats: list[Chat]) -> str:
        previous_chats = ChatService.__build_openai_rag_chat_histories(question=question, chats=chats)
        return AiService.chat_with_claude(model=model, question=context, previous_chats=previous_chats, system_prompt=SYSTEM_PROMPT)

    @staticmethod
    async def __ask_mistral_with_rag(model: str, question: str, context: str, chats: list[Chat]) -> str:
        previous_chats = ChatService.__build_openai_rag_chat_histories(question=question, chats=chats)
        return AiService.chat_with_mistral(model=model, question=context, previous_chats=previous_chats, system_prompt=SYSTEM_PROMPT)

    @staticmethod
    async def __ask_ollama_with_rag(model: str, question: str, context: str, chats: list[Chat]) -> str:
        previous_chats = ChatService.__build_openai_rag_chat_histories(question=question, chats=chats)
        return AiService.chat_with_mistral(model=model, question=context, previous_chats=previous_chats, system_prompt=SYSTEM_PROMPT)

    @staticmethod
    def get_chat_histories(video_id: int) -> list[{}]:
        """
        Retrieves the chat history for a specific video.

        This method fetches all chat interactions associated with the given video ID and
        formats them for easy consumption.

        Logic:
        1. Finds the Video object using the provided ID.
        2. Queries the database for all Chat objects associated with this video.
        3. Formats each Chat object into a dictionary, removing the video reference to avoid circular references.

        Args:
            video_id (int): The ID of the video to retrieve chat history for.

        Returns:
            list[dict]: A list of dictionaries, each representing a chat interaction.

        Note:
            The returned dictionaries are created using model_to_dict, which converts model instances to dictionaries.
        """
        selected_video: Video = VideoService.find_video_by_id(video_id)
        chat_histories = list(Chat.select().where(Chat.video == selected_video).order_by(Chat.id.asc()))
        result = []
        for chat in chat_histories:
            chat.video = None
            result.append(model_to_dict(chat))

        return result

    @staticmethod
    def clear_chat(video_id: int):
        """
        Clears all chat history associated with a specific video.

        This method deletes all Chat objects linked to the given video ID from the database.

        Logic:
        1. Finds the Video object using the provided ID.
        2. Queries the database for all Chat objects associated with this video.
        3. Iterates through each Chat object and deletes it from the database.

        Args:
            video_id (int): The ID of the video to clear chat history for.

        Note:
            This operation is irreversible and will permanently delete all chat data for the specified video.
        """
        selected_video: Video = VideoService.find_video_by_id(video_id)
        for chat in Chat.select().where(Chat.video == selected_video):
            chat.delete_instance()
