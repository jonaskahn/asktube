import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor

import iso639
from langdetect import detect

from backend.db.models import Video, VideoChapter, Chat
from backend.db.specs import sqlite_client
from backend.error.base import LogicError
from backend.error.video_error import VideoNotFoundError, VideoNotAnalyzedError
from backend.services.ai_service import AiService
from backend.utils.logger import log
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
        video_chapters = list(VideoChapter.select().where(VideoChapter.video == video))
        result: list[dict] = []
        with ThreadPoolExecutor(max_workers=len(video_chapters)) as executor:
            if provider == "gemini":
                futures = [executor.submit(VideoService.__analysis_video_with_gemini, chapter) for chapter in video_chapters]
            elif provider == "openai":
                futures = [executor.submit(VideoService.__analysis_video_with_openai, chapter) for chapter in video_chapters]
            elif provider == "voyageai":
                futures = [executor.submit(VideoService.__analysis_video_with_voyageai, chapter) for chapter in video_chapters]
            elif provider == "local":
                futures = [executor.submit(VideoService.__analysis_video_with_local, chapter) for chapter in video_chapters]
            else:
                raise LogicError("Unknown provider")
            result.extend(future.result() for future in concurrent.futures.as_completed(futures))

        ids: list[str] = []
        texts: list[str] = []
        embeddings: list[list[float]] = []
        for r in result:
            ids.append(r[0])
            texts.append(r[1])
            embeddings.append(r[2])

        AiService.store_embeddings(f"video_chapter_{video.id}", ids, texts, embeddings)
        video.is_analyzed = True
        video.embedding_provider = provider
        video.save()

    @staticmethod
    def __analysis_video_with_gemini(chapter: VideoChapter):
        return str(chapter.id), chapter.transcript, AiService.embedding_document_with_gemini(f"## {chapter.title}\n---\n{chapter.transcript}")

    @staticmethod
    def __analysis_video_with_openai(chapter: VideoChapter):
        return str(chapter.id), chapter.transcript, AiService.embedding_document_with_openai(chapter.transcript)

    @staticmethod
    def __analysis_video_with_voyageai(chapter: VideoChapter):
        return str(chapter.id), chapter.transcript, AiService.embedding_document_with_voyageai(chapter.transcript)

    @staticmethod
    def __analysis_video_with_local(chapter: VideoChapter):
        return str(chapter.id), chapter.transcript, AiService.embedding_document_with_local(chapter.transcript)

    @staticmethod
    async def summary_video(vid: int, lang_code: str, provider: str, model: str = None):
        video: Video = VideoService.find_video_by_id(vid)
        if video is None:
            raise VideoNotFoundError("Video is not found")
        if not video.is_analyzed:
            raise VideoNotAnalyzedError("Video is not analyzed")
        language = iso639.Language.from_part1(lang_code).name
        prompt = SUMMARY_PROMPT.format(**{
            "url": video.url,
            "title": video.title,
            "description": video.description,
            "transcript": video.transcript,
            "language": language,
        })

        if provider == "gemini":
            result = AiService.generate_text_with_gemini(prompt=prompt, model=model)
        elif provider == "openai":
            result = AiService.generate_text_with_openai(prompt=prompt)
        elif provider == "ollama":
            raise LogicError("Not implemented yet")
        else:
            raise LogicError("Unknown provider")
        log.debug(f"video summary: \n{result}")
        video.summary = result
        video.save()
        return result

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
        re_question = VideoService.__re_question(model, provider, question, question_lang, video)
        embedding_question = VideoService.__get_query_embedding(video.embedding_provider, re_question)
        context = AiService.query_embeddings(
            table=f"video_chapter_{video.id}",
            query=embedding_question,
            fetch_size=video.amount_chapters,
            expect_size=video.amount_chapters if video.amount_chapters <= 3 else video.amount_chapters // 2
        )
        asking_prompt = ASKING_PROMPT.format(**{
            "title": video.title,
            "question": re_question,
            "context": context,
            "language": iso639.Language.from_part1(question_lang).name
        })
        chats: list[Chat] = list(Chat.select().where(Chat.video == video))
        chat_histories = []
        for chat in chats:
            chat_histories.extend(
                (
                    {"role": "user", "parts": chat.question},
                    {"role": "model", "parts": chat.answer},
                )
            )
        if provider == "gemini":
            result = AiService.chat_with_gemini(prompt=asking_prompt, model=model, previous_chats=chat_histories)
        elif provider == "ollama":
            raise LogicError("Not implemented yet")
        elif provider == "openai":
            result = AiService.chat_with_openai(prompt=asking_prompt, model=model, previous_chats=chat_histories)
        else:
            raise LogicError("Unknown provider")
        chat = Chat.create(
            video=video,
            question=question,
            answer=result,
            context=context,
            prompt=asking_prompt
        )
        chat.save()
        return result

    @staticmethod
    def __re_question(model, provider, question, question_lang, video):
        if question_lang != video.language:
            prompt = RE_QUESTION_PROMPT.format(**{
                "video_lang": iso639.Language.from_part1(video.language).name,
                "question_lang": iso639.Language.from_part1(question_lang).name,
                "title": video.title,
                "summary": video.summary,
                "question": question
            })
            if provider == "gemini":
                question = AiService.generate_text_with_gemini(prompt=prompt, model=model)
            elif provider == "openai":
                question = AiService.generate_text_with_openai(prompt=prompt, model=model)
            elif provider == "ollama":
                raise LogicError("Not implemented yet")
            else:
                raise LogicError("Unknown provider")
        return question

    @staticmethod
    def __get_query_embedding(provider, question):
        if provider == "gemini":
            return AiService.embedding_document_with_gemini(question)
        elif provider == "openai":
            return AiService.embedding_document_with_openai(question)
        elif provider == "voyageai":
            return AiService.embedding_document_with_voyageai(question)
        elif provider == "local":
            return AiService.embedding_document_with_local(question)
        else:
            raise LogicError("Unknown provider")
