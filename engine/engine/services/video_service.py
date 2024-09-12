import asyncio
import concurrent.futures
import json
from concurrent.futures.thread import ThreadPoolExecutor

import iso639
import tiktoken
from lingua import LanguageDetectorBuilder

from engine.assistants import constants, env
from engine.assistants.errors import VideoError, AiError
from engine.assistants.logger import log
from engine.assistants.prompts import SUMMARY_PROMPT, ASKING_PROMPT, REFINED_QUESTION_PROMPT
from engine.database.models import Video, VideoChapter, Chat
from engine.database.specs import sqlite_client
from engine.services.ai_service import AiService

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
                    chapter.video = video
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
        log.debug("start analysis video")
        video: Video = VideoService.find_video_by_id(vid)
        try:
            if video is None:
                raise VideoError("video is not found")
            if video.analysis_state in [constants.ANALYSIS_STAGE_COMPLETED, constants.ANALYSIS_STAGE_PROCESSING]:
                return video

            video_chapters = VideoService.__get_video_chapters(video)

            VideoService.__update_analysis_content_state(video, constants.ANALYSIS_STAGE_PROCESSING)
            VideoService.__prepare_video_transcript(video, video_chapters)

            await VideoService.__analysis_chapters(video_chapters, provider)

            video.analysis_state = constants.ANALYSIS_STAGE_COMPLETED
            video.embedding_provider = provider
            VideoService.save(video, video_chapters)
        except Exception as e:
            VideoService.__update_analysis_content_state(video, constants.ANALYSIS_STAGE_INITIAL)
            raise e
        log.debug("finish analysis video")

    @staticmethod
    def __prepare_video_transcript(video: Video, video_chapters: list[VideoChapter]):
        if not video.raw_transcript:
            with ThreadPoolExecutor(max_workers=len(video_chapters)) as executor:
                futures = [executor.submit(AiService.speech_to_text, chapter.audio_path, chapter.start_time) for chapter in video_chapters]
            concurrent.futures.as_completed(futures)
            transcripts = []
            for future in concurrent.futures.as_completed(futures):
                transcripts.append(future.result())
            sorted_transcripts = sorted([item for sublist in transcripts for item in sublist], key=lambda x: x['start_time'])
            video.raw_transcript = json.dumps(sorted_transcripts, ensure_ascii=False)
        raw_transcripts = json.loads(video.raw_transcript) if video.raw_transcript else None
        VideoService.__pair_video_chapters_with_transcripts(video, video_chapters, raw_transcripts)
        video.transcript_tokens = len(tiktoken.get_encoding("cl100k_base").encode(video.transcript))

    @staticmethod
    def __pair_video_chapters_with_transcripts(video: Video, video_chapters: list[VideoChapter], transcripts: [{}]):
        if len(transcripts) == 0:
            raise VideoError("transcript should never be empty")
        for chapter in video_chapters:
            start_ms = chapter.start_time * 1000
            end_ms = (chapter.start_time + chapter.duration) * 1000
            chapter_transcript: str = ""
            for transcript in transcripts:
                start_transcript_ms = transcript['start_time']
                duration_transcript_ms = transcript['duration']
                if start_transcript_ms is None or start_transcript_ms <= 0 or duration_transcript_ms is None:
                    log.warn("skip this invalid transcript part")
                    continue

                end_transcript_ms = start_transcript_ms + duration_transcript_ms
                if start_transcript_ms < start_ms or end_transcript_ms > end_ms:
                    continue
                chapter_transcript += f"{transcript['text']}\n"
            if chapter_transcript != "":
                log.debug(f"title: {chapter.title}\ntranscript: {chapter_transcript}")
                chapter.transcript = chapter_transcript
        video_transcript = "\n".join([f"## {ct.title}\n-----\n{ct.transcript}" for ct in video_chapters if ct.transcript])
        video.transcript = video_transcript

    @staticmethod
    def __update_analysis_content_state(video: Video, state: int):
        with sqlite_client.atomic() as transaction:
            try:
                video.analysis_state = state
                video.save()
                transaction.commit()
            except Exception as e:
                transaction.rollback()
                raise e

    @staticmethod
    def __update_analysis_summary_state(video: Video, state: int):
        with sqlite_client.atomic() as transaction:
            try:
                video.analysis_summary_state = state
                video.save()
                transaction.commit()
            except Exception as e:
                transaction.rollback()
                raise e

    @staticmethod
    def __update_processing_stats(video: Video, is_analysis_content: bool = True, is_analysis_summary: bool = True):
        with sqlite_client.atomic() as transaction:
            try:
                if is_analysis_content:
                    video.analysis_state = constants.ANALYSIS_STAGE_PROCESSING
                if is_analysis_summary:
                    video.analysis_summary_state = constants.ANALYSIS_STAGE_PROCESSING
                video.save()
                transaction.commit()
            except Exception as e:
                transaction.rollback()
                raise e

    @staticmethod
    async def __analysis_chapters(video_chapters: list[VideoChapter], provider: str):
        """
        Analyzes video chapters using a specified provider.sqlite_master

        Args:
            provider (str): The provider for the analysis.chapter.save()
            video (Video): The video object containing the chapters to be analyzed.

        Returns:
            None
        """
        with ThreadPoolExecutor(max_workers=len(video_chapters)) as executor:
            if provider == "gemini":
                futures = [executor.submit(VideoService.__analysis_video_with_gemini, chapter) for chapter in video_chapters]
            elif provider == "openai":
                futures = [executor.submit(VideoService.__analysis_video_with_openai, chapter) for chapter in video_chapters]
            elif provider == "voyageai":
                futures = [executor.submit(VideoService.__analysis_video_with_voyageai, chapter) for chapter in video_chapters]
            elif provider == "mistral":
                futures = [executor.submit(VideoService.__analysis_video_with_mistral, chapter) for chapter in video_chapters]
            elif provider == "local":
                futures = [executor.submit(VideoService.__analysis_video_with_local, chapter) for chapter in video_chapters]
            else:
                raise AiError("unknown embedding provider")
            concurrent.futures.as_completed(futures)
            for future in concurrent.futures.as_completed(futures):
                future.result()

    @staticmethod
    def __get_video_chapters(video: Video) -> list[VideoChapter]:
        video_chapters = list(VideoChapter.select().where(VideoChapter.video == video).order_by(VideoChapter.chapter_no))
        for video_chapter in video_chapters:
            video_chapter.vid = video.id
        return video_chapters

    @staticmethod
    def __analysis_video_with_gemini(chapter: VideoChapter):
        """
        Analyzes a video chapter using the Gemini AI service.

        Args:
            chapter (VideoChapter): The chapter of the video to be analyzed.

        Returns:
            None
        """
        texts, embeddings = AiService.embedding_document_with_gemini(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, texts, embeddings)

    @staticmethod
    def __store_embedding_chunked_transcript(chapter: VideoChapter, texts: list[str], embeddings: list[list[float]]):
        """
        Stores the embedding of a video chapter transcript in a chunked manner.

        Args:
            chapter: The video chapter object containing the transcript to be stored.
            embeddings: The embeddings of the transcript.
            texts: The text chunks of the transcript.

        Returns:
            None
        """
        ids: list[str] = []
        documents: list[str] = []

        if len(texts) == 1:
            ids.append(f"{chapter.vid}_{chapter.id}")
            documents.append(f"## {chapter.title}: \n---\n{texts[0]}")
        else:
            for index, text in enumerate(texts):
                ids.append(f"{chapter.vid}_{chapter.id}_{index}")
                documents.append(f"## {chapter.title} - Part {index + 1}: \n---\n{text}")
        AiService.store_embeddings(f"video_{chapter.vid}", ids, documents, embeddings)

    @staticmethod
    def __analysis_video_with_openai(chapter: VideoChapter):
        """
        Analyzes a video chapter using the OpenAI service.

        Args:
            chapter (VideoChapter): The chapter of the video to be analyzed.

        Returns:
            None
        """
        texts, embeddings = AiService.embedding_document_with_openai(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, texts, embeddings)

    @staticmethod
    def __analysis_video_with_voyageai(chapter: VideoChapter):
        """
        Analyzes a video chapter using the VoyageAI service.

        Args:
            chapter (VideoChapter): The chapter of the video to be analyzed.

        Returns:
            None
        """
        texts, embeddings = AiService.embedding_document_with_voyageai(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, texts, embeddings)

    @staticmethod
    def __analysis_video_with_mistral(chapter: VideoChapter):
        """
        Analyzes a video chapter using the Mistral service.

        Args:
            chapter (VideoChapter): The chapter of the video to be analyzed.

        Returns:
            None
        """
        texts, embeddings = AiService.embedding_document_with_mistral(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, texts, embeddings)

    @staticmethod
    def __analysis_video_with_local(chapter: VideoChapter):
        """
        Analyzes a video chapter using a local service.

        Args:
            chapter (VideoChapter): The chapter of the video to be analyzed.

        Returns:
            None
        """
        texts, embeddings = AiService.embedding_document_with_local(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, texts, embeddings)

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
            raise VideoError("video is not found")
        video.summary = VideoService.__summary_content(lang_code, model, provider, video)
        asyncio.create_task(VideoService.__analysis_summary_video(model, provider, video))
        log.debug("finish summary video")
        return video.summary

    @staticmethod
    def __summary_content(lang_code: str, model: str, provider: str, video: Video) -> str:
        language = iso639.Language.from_part1(lang_code).name
        prompt = SUMMARY_PROMPT.format(**{
            "url": video.url,
            "title": video.title,
            "description": video.description,
            "transcript": video.transcript,
            "language": language,
        })
        return VideoService.__get_response_from_ai(model=model, prompt=prompt, provider=provider)

    @staticmethod
    async def __analysis_summary_video(model: str, provider: str, video: Video):
        log.debug("start analysis summary video")
        if video.analysis_summary_state in [constants.ANALYSIS_STAGE_COMPLETED, constants.ANALYSIS_STAGE_PROCESSING]:
            return
        VideoService.__update_processing_stats(video, is_analysis_content=False, is_analysis_summary=True)
        system_summary = VideoService.__summary_content(video.language, model, provider, video)
        texts, embeddings = VideoService.__get_query_embedding(video.embedding_provider, system_summary)
        ids: list[str] = []
        documents: list[str] = []
        for index, text in enumerate(texts):
            ids.append(f"{video.id}_0_{index}")
            documents.append(f"## Summary - Part {index + 1}: \n---\n{text}")
        AiService.store_embeddings(f"video_summary_{video.id}", ids, texts, embeddings)
        video.analysis_summary_state = constants.ANALYSIS_STAGE_COMPLETED
        video.save()
        log.debug("finish analysis summary video")

    @staticmethod
    def __get_query_embedding(provider: str, text: str) -> tuple[list[str], list[list[float]]]:
        if provider == "gemini":
            return AiService.embedding_document_with_gemini(text)
        elif provider == "openai":
            return AiService.embedding_document_with_openai(text)
        elif provider == "voyageai":
            return AiService.embedding_document_with_voyageai(text)
        elif provider == "mistral":
            return AiService.embedding_document_with_mistral(text)
        elif provider == "local":
            return AiService.embedding_document_with_local(text)
        else:
            raise AiError("unknown embedding provider")

    @staticmethod
    async def ask(question: str, vid: int, provider: str, model: str = None):
        """
        Asks a question about a video and returns the answer.

        Since question and video may not same language, a small call to AI Provider will be trigger 
        to translate question if needed. In the next step, refined question will be embedding and
        query compare in the ChromaDB to find the similar transcript. Finally, when we have enough
        information to enrich the question, we will ask it to AI Provider to get the answer.


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
            raise VideoError("video is not found")
        if not video.analysis_state:
            raise VideoError("video has not analyzed yet")
        question_lang = detector.detect_language_of(question)
        question_lang_code = question_lang.iso_code_639_1.name.__str__().lower()
        chats: list[Chat] = list(Chat.select().where(Chat.video == video).limit(10))
        refined_question = VideoService.__refine_question(model, provider, question, question_lang_code, video)
        _, embedding_question = VideoService.__get_query_embedding(video.embedding_provider, refined_question)

        amount, context = AiService.query_embeddings(
            table=f"video_{video.id}",
            query=embedding_question,
            fetch_size=video.amount_chapters * 3
        )
        aware_context = context
        if not context:
            aware_context = "No information" if env.TOKEN_CONTEXT_THRESHOLD > video.transcript_tokens else video.transcript

        asking_prompt = ASKING_PROMPT.format(**{
            "url": video.url,
            "title": video.title,
            "context": aware_context,
            "question": question,
            "refined_question": refined_question,
            "language": question_lang.name.__str__()
        })
        result = VideoService.__get_response_from_ai(
            model=model,
            prompt=asking_prompt,
            provider=provider,
            chats=chats
        )
        chat = Chat.create(
            video=video,
            question=question,
            refined_question=refined_question,
            answer=result,
            context=context,
            prompt=asking_prompt,
            provider=provider
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
        return VideoService.__get_response_from_ai(model=model, prompt=prompt, provider=provider)

    @staticmethod
    def __get_response_from_ai(
            model: str,
            prompt: str,
            provider: str,
            chats: list[Chat] = None
    ) -> str:
        if chats is None:
            chats = []
        if provider == "gemini":
            return AiService.chat_with_gemini(model=model, prompt=prompt, previous_chats=chats)
        elif provider == "openai":
            return AiService.chat_with_openai(model=model, prompt=prompt, previous_chats=chats)
        elif provider == "claude":
            return AiService.chat_with_claude(model=model, prompt=prompt, previous_chats=chats)
        elif provider == "mistral":
            return AiService.chat_with_mistral(model=model, prompt=prompt, previous_chats=chats)
        elif provider == "ollama":
            return AiService.chat_with_ollama(model=model, prompt=prompt, previous_chats=chats)
        else:
            raise AiError("unknown AI provider")

    @staticmethod
    def delete(video_id: int):

        with sqlite_client.atomic() as transaction:
            try:
                video = VideoService.find_video_by_id(video_id)
                chapters = list(VideoChapter.select().where(VideoChapter.video == video))
                video.delete_instance()
                for chapter in chapters:
                    chapter.delete_instance()
                AiService.delete_collection(f"video_{video_id}")
                transaction.commit()
            except Exception as e:
                transaction.rollback()
                raise e
