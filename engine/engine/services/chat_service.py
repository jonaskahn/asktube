from playhouse.shortcuts import model_to_dict

from engine.database.models import Video, Chat
from engine.services.ai_service import AiService
from engine.services.video_service import VideoService
from engine.supports import env
from engine.supports.errors import ChatError
from engine.supports.prompts import ASKING_PROMPT


class ChatService:
    def __init__(self):
        pass

    @staticmethod
    async def ask(question: str, vid: int, provider: str, model: str = None) -> dict[object]:
        video: Video = VideoService.get_analyzed_video(vid)
        chats: list[Chat] = list(Chat.select().where(Chat.video == video).limit(20).order_by(Chat.id.asc()))
        if video.transcript_tokens <= env.TOKEN_CONTEXT_THRESHOLD:
            return await ChatService.__ask_without_rag(question, video, chats, provider, model)
        else:
            return await ChatService.__ask_with_rag(question, video, chats, provider, model)

    @staticmethod
    async def __ask_without_rag(question: str, video: Video, chats: list[Chat], provider: str, model: str = None) -> dict[object]:
        pass

    @staticmethod
    async def __ask_with_rag(question: str, video: Video, chats: list[Chat], provider: str, model: str = None) -> dict[object]:
        _, embedding_question = AiService.get_texts_embedding(video.embedding_provider, question)

        amount, document = AiService.query_embeddings(
            table=f"video_{video.id}",
            query=embedding_question,
            fetch_size=video.total_parts
        )

        context = ASKING_PROMPT.format(**{
            "title": video.title,
            "url": video.url,
            "context": document
        }) if document else "No information, just answer me if you have ability"
        match provider:
            case "gemini":
                result = await ChatService.__ask_gemini_with_rag(model=model, question=question, context=context, chats=chats)
            case "openai":
                result = await ChatService.__ask_openai_with_rag(model=model, question=question, context=context, chats=chats)
            case "claude":
                result = await ChatService.__ask_claude_with_rag(model=model, question=question, context=context, chats=chats)
            case "mistral":
                result = await ChatService.__ask_mistral_with_rag(model=model, question=question, context=context, chats=chats)
            case "ollama":
                result = await ChatService.__ask_ollama_with_rag(model=model, question=question, context=context, chats=chats)
            case _:
                raise ChatError("unknown chat provider")

        chat = Chat.create(
            video=video,
            question=question,
            refined_question="Not Implemented Yet",
            answer=result,
            context=document if document else "",
            prompt=context if context else "",
            provider=provider
        )
        chat.save()
        return model_to_dict(chat)

    @staticmethod
    async def __ask_gemini_with_rag(model: str, question: str, context: str, chats: list[Chat]) -> str:
        previous_chats = ChatService.__build_gemini_rag_chat_histories(question=question, chats=chats)
        return AiService.chat_with_gemini(model=model, question=context, previous_chats=previous_chats)

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
        return AiService.chat_with_openai(model=model, question=context, previous_chats=previous_chats)

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
        return AiService.chat_with_claude(model=model, question=context, previous_chats=previous_chats)

    @staticmethod
    async def __ask_mistral_with_rag(model: str, question: str, context: str, chats: list[Chat]) -> str:
        previous_chats = ChatService.__build_openai_rag_chat_histories(question=question, chats=chats)
        return AiService.chat_with_mistral(model=model, question=context, previous_chats=previous_chats)

    @staticmethod
    async def __ask_ollama_with_rag(model: str, question: str, context: str, chats: list[Chat]) -> str:
        previous_chats = ChatService.__build_openai_rag_chat_histories(question=question, chats=chats)
        return AiService.chat_with_mistral(model=model, question=context, previous_chats=previous_chats)

    @staticmethod
    def get_chat_histories(video_id: int) -> list[{}]:
        selected_video: Video = VideoService.find_video_by_id(video_id)
        chat_histories = list(Chat.select().where(Chat.video == selected_video).order_by(Chat.id.asc()))
        result = []
        for chat in chat_histories:
            chat.video = None
            result.append(model_to_dict(chat))

        return result

    @staticmethod
    def clear_chat(video_id: int):
        selected_video: Video = VideoService.find_video_by_id(video_id)
        for chat in Chat.select().where(Chat.video == selected_video):
            chat.delete_instance()
