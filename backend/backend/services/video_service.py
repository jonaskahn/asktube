import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor

import iso639
from langdetect import detect

from backend.db.models import Video, VideoChapter, Chat
from backend.db.specs import sqlite_client
from backend.error.base import LogicError
from backend.error.video_error import VideoNotFoundError, VideoNotAnalyzedError
from backend.services.ai_service import AiService
from backend.utils.prompts import SUMMARY_PROMPT, RE_QUESTION_PROMPT, ASKING_PROMPT


class VideoService:
    def __init__(self):
        self.__ai_service = AiService()

    @staticmethod
    def save(video: Video, chapters: list[VideoChapter]):
        with sqlite_client.atomic() as transaction:
            try:
                video.save()
                for chapter in chapters:
                    chapter.save()
                transaction.commit()
            except Exception as e:
                transaction.rollback()
                raise e

    @staticmethod
    def find_video_by_youtube_id(youtube_id: str):
        return Video.get_or_none(Video.youtube_id == youtube_id)

    @staticmethod
    def find_video_by_id(vid: int):
        return Video.get_or_none(Video.id == vid)

    @staticmethod
    async def analysis_video(vid: int, provider: str = "gemini"):
        video: Video = VideoService.find_video_by_id(vid)
        if video is None:
            raise VideoNotFoundError("Video not found")
        if video.is_analyzed:
            return video
        await VideoService.__analysis_chapters(provider, video)
        video.is_analyzed = True
        video.embedding_provider = provider
        video.save()

    @staticmethod
    async def __analysis_chapters(provider, video):
        video_chapters = list(VideoChapter.select().where(VideoChapter.video == video))
        with ThreadPoolExecutor(max_workers=len(video_chapters)) as executor:
            if provider == "gemini":
                futures = [executor.submit(VideoService.__analysis_video_with_gemini, video.id, chapter) for chapter in video_chapters]
            elif provider == "openai":
                futures = [executor.submit(VideoService.__analysis_video_with_openai, video.id, chapter) for chapter in video_chapters]
            elif provider == "voyageai":
                futures = [executor.submit(VideoService.__analysis_video_with_voyageai, video.id, chapter) for chapter in video_chapters]
            elif provider == "local":
                futures = [executor.submit(VideoService.__analysis_video_with_local, video.id, chapter) for chapter in video_chapters]
            else:
                raise LogicError("Unknown provider")
            concurrent.futures.as_completed(futures)
            for future in concurrent.futures.as_completed(futures):
                future.result()

    @staticmethod
    def __analysis_video_with_gemini(vid: int, chapter: VideoChapter):
        texts, embeddings = AiService.embedding_document_with_gemini(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, embeddings, texts, vid)

    @staticmethod
    def __store_embedding_chunked_transcript(chapter, embeddings, texts, vid):
        ids: list[str] = []
        documents: list[str] = []

        if len(texts) == 1:
            ids.append(f"{vid}_{chapter.id}")
            documents.append(f"## {chapter.title}: \n---\n{texts[0]}")
        else:
            for index, text in enumerate(texts):
                ids.append(f"{vid}_{chapter.id}_{index}")
                documents.append(f"## {chapter.title} - Part {index + 1}: \n---\n{text}")
        AiService.store_embeddings(f"video_chapter_{vid}", ids, documents, embeddings)

    @staticmethod
    def __analysis_video_with_openai(vid: int, chapter: VideoChapter):
        texts, embeddings = AiService.embedding_document_with_openai(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, embeddings, texts, vid)

    @staticmethod
    def __analysis_video_with_voyageai(vid: int, chapter: VideoChapter):
        texts, embeddings = AiService.embedding_document_with_voyageai(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, embeddings, texts, vid)

    @staticmethod
    def __analysis_video_with_local(vid: int, chapter: VideoChapter):
        texts, embeddings = AiService.embedding_document_with_local(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, embeddings, texts, vid)

    @staticmethod
    async def summary_video(vid: int, lang_code: str, provider: str, model: str = None):
        video: Video = VideoService.find_video_by_id(vid)
        if video is None:
            raise VideoNotFoundError("Video is not found")
        if not video.is_analyzed:
            raise VideoNotAnalyzedError("Video is not analyzed")
        user_summary = await VideoService.__summary_content(lang_code, model, provider, video)
        await VideoService.__analysis_summary_video(model, provider, vid, video)
        video.summary = user_summary
        video.is_summary_analyzed = True
        video.save()
        return user_summary

    @staticmethod
    async def __analysis_summary_video(model, provider, vid, video):
        if video.is_summary_analyzed:
            return
        system_summary = await VideoService.__summary_content(video.language, model, provider, video)
        texts, embeddings = VideoService.__get_query_embedding(video.embedding_provider, system_summary)
        AiService.store_embeddings(f"video_summary_{vid}", [str(0)], texts, embeddings)

    @staticmethod
    async def __summary_content(lang_code, model, provider, video):
        language = iso639.Language.from_part1(lang_code).name
        prompt = SUMMARY_PROMPT.format(**{
            "url": video.url,
            "title": video.title,
            "description": video.description,
            "transcript": video.transcript,
            "language": language,
        })
        return VideoService.__get_response_from_ai(prompt=prompt, model=model, provider=provider)

    @staticmethod
    async def ask(question: str, vid: int, provider: str, model: str = None):
        video: Video = VideoService.find_video_by_id(vid)
        if video is None:
            raise VideoNotFoundError("Video is not found")
        if not video.is_analyzed:
            raise VideoNotAnalyzedError("Video is not analyzed")
        question_lang = detect(question)
        if video.summary is None:
            await VideoService.summary_video(vid, question_lang, provider, model)
        chats: list[Chat] = list(Chat.select().where(Chat.video == video).limit(10))
        refined_question = VideoService.__refine_question(model, provider, question, question_lang, video, chats)
        _, embedding_question = VideoService.__get_query_embedding(video.embedding_provider, refined_question)
        amount, context = AiService.query_embeddings(
            table=f"video_chapter_{video.id}",
            query=embedding_question,
            fetch_size=video.amount_chapters
        )
        asking_prompt = ASKING_PROMPT.format(**{
            "title": video.title,
            "question": refined_question,
            "context": context,
            "language": iso639.Language.from_part1(question_lang).name
        })
        result = VideoService.__get_response_from_ai(prompt=asking_prompt, model=model, provider=provider, chats=chats)
        chat = Chat.create(
            video=video,
            question=question,
            refined_question=refined_question,
            answer=result,
            context=context,
            prompt=asking_prompt
        )
        chat.save()
        return result

    @staticmethod
    def __refine_question(model: str, provider: str, question: str, question_lang: str, video: Video, chats: list[Chat]):
        prompt = RE_QUESTION_PROMPT.format(**{
            "video_lang": iso639.Language.from_part1(video.language).name,
            "question_lang": iso639.Language.from_part1(question_lang).name,
            "title": video.title,
            "summary": video.summary,
            "question": question
        })
        return VideoService.__get_response_from_ai(prompt=prompt, model=model, provider=provider, chats=chats)

    @staticmethod
    def __get_response_from_ai(prompt: str, model: str, provider: str, chats: list[Chat] = None):
        if chats is None:
            chats = []
        if provider == "gemini":
            return AiService.chat_with_gemini(prompt=prompt, model=model, previous_chats=chats)
        elif provider == "openai":
            return AiService.chat_with_openai(prompt=prompt, model=model, previous_chats=chats)
        elif provider == "claude":
            return AiService.chat_with_claude(prompt=prompt, model=model, previous_chats=chats)
        elif provider == "ollama":
            raise AiService.chat_with_ollama(prompt=prompt, model=model, previous_chats=chats)
        else:
            raise LogicError("Unknown provider")

    @staticmethod
    def __get_query_embedding(provider: str, text: str):
        if provider == "gemini":
            return AiService.embedding_document_with_gemini(text)
        elif provider == "openai":
            return AiService.embedding_document_with_openai(text)
        elif provider == "voyageai":
            return AiService.embedding_document_with_voyageai(text)
        elif provider == "local":
            return AiService.embedding_document_with_local(text)
        else:
            raise LogicError("Unknown provider")
