import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor

import iso639
from backend.db.models import Video, VideoChapter, Chat
from backend.db.specs import sqlite_client
from backend.error.base import LogicError
from backend.error.video_error import VideoNotFoundError, VideoNotAnalyzedError
from backend.services.ai_service import AiService
from backend.utils.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT, ASKING_PROMPT
from lingua import LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_all_languages().build()


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
        """
        Analyzes a video by its ID and updates its analysis status.

        Fetch all chapter transcripts of video then embedding them with the specified provider.
        After that, all embeddings will be stored in the database.

        Args:
            vid (int): The ID of the video to be analyzed.
            provider (str, optional): The provider for the analysis. Defaults to "gemini".

        Returns:
            Video: The analyzed video object.
        """
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
        """
        Analyzes video chapters using a specified provider.

        Args:
            provider (str): The provider for the analysis.
            video (Video): The video object containing the chapters to be analyzed.

        Returns:
            None
        """
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
        """
        Analyzes a video chapter using the Gemini AI service.

        Args:
            vid (int): The ID of the video.
            chapter (VideoChapter): The chapter of the video to be analyzed.

        Returns:
            None
        """
        texts, embeddings = AiService.embedding_document_with_gemini(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, embeddings, texts, vid)

    @staticmethod
    def __store_embedding_chunked_transcript(chapter, embeddings, texts, vid):
        """
        Stores the embedding of a video chapter transcript in a chunked manner.

        Args:
            chapter: The video chapter object containing the transcript to be stored.
            embeddings: The embeddings of the transcript.
            texts: The text chunks of the transcript.
            vid: The ID of the video.

        Returns:
            None
        """
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
        """
        Analyzes a video chapter using the OpenAI service.

        Args:
            vid (int): The ID of the video.
            chapter (VideoChapter): The chapter of the video to be analyzed.

        Returns:
            None
        """
        texts, embeddings = AiService.embedding_document_with_openai(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, embeddings, texts, vid)

    @staticmethod
    def __analysis_video_with_voyageai(vid: int, chapter: VideoChapter):
        """
        Analyzes a video chapter using the VoyageAI service.

        Args:
            vid (int): The ID of the video.
            chapter (VideoChapter): The chapter of the video to be analyzed.

        Returns:
            None
        """
        texts, embeddings = AiService.embedding_document_with_voyageai(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, embeddings, texts, vid)

    @staticmethod
    def __analysis_video_with_local(vid: int, chapter: VideoChapter):
        """
        Analyzes a video chapter using a local service.

        Args:
            vid (int): The ID of the video.
            chapter (VideoChapter): The chapter of the video to be analyzed.

        Returns:
            None
        """
        texts, embeddings = AiService.embedding_document_with_local(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, embeddings, texts, vid)

    @staticmethod
    async def summary_video(vid: int, lang_code: str, provider: str, model: str = None):
        """
        Retrieves the summary of a video using transcript.

        After generate requested summary, a "system summary" will be generated,
        then embedded using Provider same as video chapters and store to ChromaDB.

        Args:
            vid (int): The ID of the video.
            lang_code (str): The language code of the video.
            provider (str): The provider of the video.
            model (str, optional): The model to use for summarization. Defaults to None.

        Returns:
            str: The summary of the video.

        Raises:
            VideoNotFoundError: If the video is not found.
            VideoNotAnalyzedError: If the video is not analyzed.

        """
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
        """
        Asks a question about a video and returns the answer.

        Since question adn video chapter transcript may not same language, a small call to AI Provider
        will be trigger to translate question if needed. In the next step, refined question will be
        embbeding and query compare in the ChromaDB to find the similiar transcript. Finally, when we
        have enough information to enrich the question, we will ask it to AI Provider to get the answer.


        Args:
            question (str): The question to ask about the video.
            vid (int): The ID of the video.
            provider (str): The provider of the video.
            model (str, optional): The model to use for answering the question. Defaults to None.

        Returns:
            str: The answer to the question.

        Raises:
            VideoNotFoundError: If the video is not found.
            VideoNotAnalyzedError: If the video is not analyzed.
        """
        video: Video = VideoService.find_video_by_id(vid)
        if video is None:
            raise VideoNotFoundError("Video is not found")
        if not video.is_analyzed:
            raise VideoNotAnalyzedError("Video is not analyzed")
        question_lang = detector.detect_language_of(question)
        question_lang_code = question_lang.iso_code_639_1.name.__str__().lower()
        if video.summary is None:
            await VideoService.summary_video(vid, question_lang_code, provider, model)
        chats: list[Chat] = list(Chat.select().where(Chat.video == video).limit(10))
        refined_question = VideoService.__refine_question(model, provider, question, question_lang_code, video)
        _, embedding_question = VideoService.__get_query_embedding(video.embedding_provider, refined_question)
        amount, context = AiService.query_embeddings(
            table=f"video_chapter_{video.id}",
            query=embedding_question,
            fetch_size=video.amount_chapters * 3
        )
        asking_prompt = ASKING_PROMPT.format(**{
            "url": video.url,
            "title": video.title,
            "context": context,
            "question": question,
            "refined_question": refined_question,
            "language": question_lang.name.__str__()
        })
        result = VideoService.__get_response_from_ai(
            prompt=asking_prompt,
            model=model,
            provider=provider,
            chats=chats
        )
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
    def __refine_question(model: str, provider: str, question: str, question_lang: str, video: Video):
        if question_lang == video.language:
            return question
        prompt = REFINED_QUESTION_PROMPT.format(**{
            "video_lang": iso639.Language.from_part1(video.language).name,
            "question": question
        })
        return VideoService.__get_response_from_ai(prompt=prompt, model=model, provider=provider)

    @staticmethod
    def __get_response_from_ai(
            prompt: str,
            system_prompt: str = SYSTEM_PROMPT,
            model: str = None,
            provider: str = None,
            chats: list[Chat] = None
    ):
        if chats is None:
            chats = []
        if provider == "gemini":
            return AiService.chat_with_gemini(prompt=prompt, system_prompt=system_prompt, model=model, previous_chats=chats)
        elif provider == "openai":
            return AiService.chat_with_openai(prompt=prompt, system_prompt=system_prompt, model=model, previous_chats=chats)
        elif provider == "claude":
            return AiService.chat_with_claude(prompt=prompt, system_prompt=system_prompt, model=model, previous_chats=chats)
        elif provider == "ollama":
            raise AiService.chat_with_ollama(prompt=prompt, system_prompt=system_prompt, model=model, previous_chats=chats)
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
