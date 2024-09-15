import asyncio
import concurrent.futures
import json
from collections import Counter
from concurrent.futures.thread import ThreadPoolExecutor

import iso639
import tiktoken
from lingua import LanguageDetectorBuilder
from playhouse.shortcuts import model_to_dict

from engine.database.models import Video, VideoChapter, Chat
from engine.database.specs import sqlite_client
from engine.services.ai_service import AiService
from engine.supports import constants, env
from engine.supports.errors import VideoError, AiError
from engine.supports.logger import log
from engine.supports.prompts import SUMMARY_PROMPT, ASKING_PROMPT, REFINED_QUESTION_PROMPT

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
    def analysis_video(vid: int):
        """
        Analyzes a video by its ID and updates its analysis status.

        Fetches all chapter transcripts of the video and embeds them using the specified provider.
        After that, all embeddings are stored in the database.

        Args:
            vid (int): The ID of the video to be analyzed.

        Returns:
            Video: The analyzed video object.

        Raises:
            VideoError: If the video is not found.
            Exception: If an error occurs during the analysis process.
        """
        video: Video = VideoService.find_video_by_id(vid)
        try:
            if video is None:
                raise VideoError("video is not found")
            if video.analysis_state in [constants.ANALYSIS_STAGE_COMPLETED, constants.ANALYSIS_STAGE_PROCESSING]:
                return video
            asyncio.create_task(VideoService.__internal_analysis(video))
            return video
        except Exception as e:
            VideoService.__update_analysis_content_state(video, constants.ANALYSIS_STAGE_INITIAL)
            raise e

    @staticmethod
    async def __internal_analysis(video):
        log.debug("start analysis video")
        video_chapters = VideoService.__get_video_chapters(video)
        VideoService.__update_analysis_content_state(video, constants.ANALYSIS_STAGE_PROCESSING)
        VideoService.__prepare_video_transcript(video, video_chapters)
        video.total_parts = await VideoService.__analysis_chapters(video_chapters, video.embedding_provider)
        video.analysis_state = constants.ANALYSIS_STAGE_COMPLETED
        VideoService.save(video, video_chapters)
        log.debug("finish analysis video")

    @staticmethod
    def __prepare_video_transcript(video: Video, video_chapters: list[VideoChapter]):
        """
        Prepares the transcript for a given video.

        Args:
            video (Video): The video object.
            video_chapters (list[VideoChapter]): The list of video chapters.

        Returns:
            None

        Raises:
            None

        This function checks if the video does not have a raw transcript. If it doesn't, it starts the process of recognizing the transcript. It uses a ThreadPoolExecutor with a maximum of 4 workers to submit speech-to-text tasks for each chapter in the video. The results of these tasks are stored in transcripts and predict_langs. Once all the tasks are completed, the transcripts are sorted based on the start_time and stored in sorted_transcripts. The function then sets the raw_transcript of the video to the sorted_transcripts in JSON format. It also determines the most common language and sets the language of the video accordingly. Finally, the function pairs the video chapters with the transcripts, calculates the number of transcript tokens, and sets the transcript_tokens attribute of the video.

        Note:
        - This function assumes that the AiService.speech_to_text function is defined and returns a tuple of the predicted language and the transcript.
        - The tiktoken.get_encoding function is assumed to be defined and returns an encoding object.
        - The constants.ANALYSIS_STAGE_INITIAL, constants.ANALYSIS_STAGE_PROCESSING, and constants.ANALYSIS_STAGE_COMPLETED are assumed to be defined.
        """

        if not video.raw_transcript:
            log.debug("start to recognize video transcript")
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(AiService.speech_to_text, chapter.audio_path, chapter.start_time) for chapter in video_chapters]
            transcripts = []
            predict_langs = []
            for future in concurrent.futures.as_completed(futures):
                lang, transcript = future.result()
                transcripts.append(transcript)
                predict_langs.append(lang)
            sorted_transcripts = sorted([item for sublist in transcripts for item in sublist], key=lambda x: x['start_time'])
            log.debug("finish to recognize video transcript")
            video.raw_transcript = json.dumps(sorted_transcripts, ensure_ascii=False)
            predict_lang, count = Counter(predict_langs).most_common(1)[0]
            video.language = predict_lang if count >= 2 or len(predict_langs) == 1 else None
        raw_transcripts = json.loads(video.raw_transcript) if video.raw_transcript else None
        VideoService.__pair_video_chapters_with_transcripts(video, video_chapters, raw_transcripts)
        video.transcript_tokens = len(tiktoken.get_encoding("cl100k_base").encode(video.transcript))

    @staticmethod
    def __pair_video_chapters_with_transcripts(video: Video, video_chapters: list[VideoChapter], transcripts: [{}]):
        """
        Pairs video chapters with their corresponding transcripts.

        Args:
            video (Video): The video object containing the chapters.
            video_chapters (list[VideoChapter]): A list of video chapters.
            transcripts ([{}]): A list of transcript dictionaries.

        Raises:
            VideoError: If the transcripts list is empty.

        Returns:
            None
        """

        if len(transcripts) == 0:
            raise VideoError("transcript should never be empty")
        for chapter in video_chapters:
            start_ms = chapter.start_time * 1000
            end_ms = (chapter.start_time + chapter.duration) * 1000
            chapter_transcript: str = ""
            for transcript in transcripts:
                start_transcript_ms = transcript['start_time']
                duration_transcript_ms = transcript['duration']
                if start_transcript_ms is None or start_transcript_ms < 0 or duration_transcript_ms is None or duration_transcript_ms < 0:
                    log.warn("skip this invalid transcript part")
                    continue

                end_transcript_ms = start_transcript_ms + duration_transcript_ms
                if start_transcript_ms < start_ms or end_transcript_ms > end_ms:
                    continue
                chapter_transcript += f"{transcript['text']}\n"
            if chapter_transcript != "":
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
    async def __analysis_chapters(video_chapters: list[VideoChapter], provider: str) -> int:
        """
        Analyzes video chapters using a specified provider.

        Args:
            video_chapters (list[VideoChapter]): A list of video chapters to be analyzed.
            provider (str): The provider to use for analysis. Can be one of "gemini", "openai", "voyageai", "mistral", or "local".

        Returns:
            int: The total number of parts analyzed.

        Raises:
            AiError: If the provider is unknown.
        """

        with ThreadPoolExecutor(max_workers=len(video_chapters)) as executor:
            if provider == "gemini":
                futures = [executor.submit(VideoService.__analysis_video_with_gemini, chapter) for chapter in video_chapters]
            elif provider == "local":
                futures = [executor.submit(VideoService.__analysis_video_with_local, chapter) for chapter in video_chapters]
            elif provider == "mistral":
                futures = [executor.submit(VideoService.__analysis_video_with_mistral, chapter) for chapter in video_chapters]
            elif provider == "openai":
                futures = [executor.submit(VideoService.__analysis_video_with_openai, chapter) for chapter in video_chapters]
            elif provider == "voyageai":
                futures = [executor.submit(VideoService.__analysis_video_with_voyageai, chapter) for chapter in video_chapters]
            else:
                log.debug(f"selected provider: {provider}")
                raise AiError("unknown embedding provider")
        return sum(
            future.result() for future in concurrent.futures.as_completed(futures)
        )

    @staticmethod
    def __get_video_chapters(video: Video) -> list[VideoChapter]:
        video_chapters = list(VideoChapter.select().where(VideoChapter.video == video).order_by(VideoChapter.chapter_no))
        for video_chapter in video_chapters:
            video_chapter.vid = video.id
        return video_chapters

    @staticmethod
    def __analysis_video_with_gemini(chapter: VideoChapter) -> int:
        texts, embeddings = AiService.embedding_document_with_gemini(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, texts, embeddings)
        return len(texts)

    @staticmethod
    def __store_embedding_chunked_transcript(chapter: VideoChapter, texts: list[str], embeddings: list[list[float]]):
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
    def __analysis_video_with_openai(chapter: VideoChapter) -> int:
        texts, embeddings = AiService.embed_document_with_openai(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, texts, embeddings)
        return len(texts)

    @staticmethod
    def __analysis_video_with_voyageai(chapter: VideoChapter):
        texts, embeddings = AiService.embed_document_with_voyageai(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, texts, embeddings)
        return len(texts)

    @staticmethod
    def __analysis_video_with_mistral(chapter: VideoChapter):
        texts, embeddings = AiService.embed_document_with_mistral(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, texts, embeddings)
        return len(texts)

    @staticmethod
    def __analysis_video_with_local(chapter: VideoChapter):
        texts, embeddings = AiService.embed_document_with_local(chapter.transcript)
        VideoService.__store_embedding_chunked_transcript(chapter, texts, embeddings)
        return len(texts)

    @staticmethod
    async def summary_video(vid: int, lang_code: str, provider: str, model: str = None):
        """
        Generates a summary of a video.

        Args:
            vid (int): The ID of the video.
            lang_code (str): The language code of the summary.
            provider (str): The provider of the video.
            model (str, optional): The model to use for generating the summary. Defaults to None.

        Returns:
            str: The summary of the video.

        Raises:
            VideoError: If the video is not found.
        """

        video: Video = VideoService.find_video_by_id(vid)
        if video is None:
            raise VideoError("video is not found")
        video.summary = VideoService.__summary_content(lang_code, model, provider, video)
        video.save()
        asyncio.create_task(VideoService.__analysis_summary_video(model, provider, video))
        log.debug("finish summary video")
        return video

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
        if video.analysis_summary_state in [constants.ANALYSIS_STAGE_COMPLETED, constants.ANALYSIS_STAGE_PROCESSING]:
            return
        log.debug("start analysis summary video")
        VideoService.__update_analysis_summary_state(video, constants.ANALYSIS_STAGE_PROCESSING)
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
    async def ask(question: str, vid: int, provider: str, model: str = None):
        """
        Asks a question about a video and returns the answer.

        The function takes a question, video id, provider, and optional model as input.
        It first checks if the video exists and has been analyzed. Then, it detects the language of the question,
        refines the question if necessary, and gets the query embedding. It queries the embeddings in the ChromaDB
        to find similar transcripts and uses the context to ask the AI provider.
        The function returns the answer from the AI provider and saves the chat history.

        Parameters:
            question (str): The question to ask about the video.
            vid (int): The id of the video.
            provider (str): The provider to use for asking the question.
            model (str): The optional model to use for asking the question.

        Returns:
            str: The answer from the AI provider.
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
            fetch_size=video.total_parts
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
        return model_to_dict(chat)

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
        """
        Deletes a video by its ID.

        This function deletes a video from the database and also removes its associated chapters.
        It uses a transaction to ensure that either all changes are committed or none are,
        in case of an exception.

        Parameters:
            video_id (int): The ID of the video to be deleted.

        Returns:
            None

        Raises:
            Exception: If any error occurs during the deletion process.
        """

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

    @staticmethod
    def get(page_no: int) -> tuple[int, list[Video]]:
        total = Video.select().count()
        limit = 48
        if total // limit < page_no:
            page_no = total // limit
        offset = (page_no - 1) * 48
        videos = list(Video.select().order_by(Video.id.desc()).offset(offset).limit(limit))
        video_data = [model_to_dict(video) for video in videos]
        return total, video_data

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
