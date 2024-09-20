import asyncio
import concurrent.futures
import json
import uuid
from collections import Counter
from concurrent.futures.thread import ThreadPoolExecutor

import iso639
import tiktoken
from playhouse.shortcuts import model_to_dict
from sanic.log import logger

from engine.database.models import Video, VideoChapter
from engine.database.specs import sqlite_client, chromadb_client
from engine.services.ai_service import AiService
from engine.supports import constants
from engine.supports.errors import VideoError, AiError
from engine.supports.prompts import SUMMARY_PROMPT


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
    def get_video_detail(vid: int):
        video = Video.get_or_none(Video.id == vid)
        if video is None:
            raise VideoError("video is not found")
        return model_to_dict(video)

    @staticmethod
    def get_analyzed_video(vid: int) -> Video:
        video: Video = VideoService.find_video_by_id(vid)
        if video is None:
            raise VideoError("video is not found")
        if not video.analysis_state:
            raise VideoError("video has not analyzed yet")
        return video

    @staticmethod
    async def analysis_video(vid: int):
        video: Video = VideoService.find_video_by_id(vid)
        if video is None:
            raise VideoError("video is not found")
        asyncio.create_task(VideoService.__internal_analysis(video))
        return model_to_dict(video)

    @staticmethod
    async def __internal_analysis(video):
        if video.analysis_state in [constants.ANALYSIS_STAGE_COMPLETED, constants.ANALYSIS_STAGE_PROCESSING]:
            return
        try:
            trace_id = uuid.uuid4()
            logger.debug(f"[{trace_id}] start analysis video: {video.title}")

            logger.debug(f"[{trace_id}] start get video chapters")
            video_chapters = VideoService.__get_video_chapters(video)
            logger.debug(f"[{trace_id}] finish get video chapters")

            VideoService.__update_analysis_content_state(video, constants.ANALYSIS_STAGE_PROCESSING)

            logger.debug(f"[{trace_id}] start prepare video chapters")
            VideoService.__prepare_video_transcript(video, video_chapters)
            logger.debug(f"[{trace_id}] finish prepare video chapters")

            logger.debug(f"[{trace_id}] start embedding video transcript")
            video.total_parts = await VideoService.__analysis_chapters(video_chapters, video.embedding_provider)
            video.analysis_state = constants.ANALYSIS_STAGE_COMPLETED
            VideoService.save(video, video_chapters)
            logger.debug(f"[{trace_id}] finish embedding video transcript")
            logger.debug(f"finish analysis video: {video.title}")
        except Exception as e:
            VideoService.__update_analysis_content_state(video, constants.ANALYSIS_STAGE_INITIAL)
            raise e

    @staticmethod
    def __prepare_video_transcript(video: Video, video_chapters: list[VideoChapter]):
        if not video.raw_transcript:
            logger.debug("start to recognize video transcript")
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(AiService.speech_to_text, chapter.audio_path, chapter.start_time) for chapter
                           in video_chapters]
            transcripts = []
            predict_langs = []
            for future in concurrent.futures.as_completed(futures):
                lang, transcript = future.result()
                transcripts.append(transcript)
                predict_langs.append(lang)
            sorted_transcripts = sorted([item for sublist in transcripts for item in sublist],
                                        key=lambda x: x['start_time'])
            logger.debug("finish to recognize video transcript")
            video.raw_transcript = json.dumps(sorted_transcripts, ensure_ascii=False)
            predict_lang, count = Counter(predict_langs).most_common(1)[0]
            video.language = predict_lang if count >= 2 or len(predict_langs) == 1 else None
        raw_transcripts = json.loads(video.raw_transcript) if video.raw_transcript else None
        VideoService.__merge_transcript_to_chapter(video, video_chapters, raw_transcripts)
        video.transcript_tokens = len(tiktoken.get_encoding("cl100k_base").encode(video.transcript))

    @staticmethod
    def __merge_transcript_to_chapter(video: Video, video_chapters: list[VideoChapter], transcripts: [{}]):
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
                    logger.warn("skip this invalid transcript part")
                    continue

                end_transcript_ms = start_transcript_ms + duration_transcript_ms
                if start_transcript_ms < start_ms or end_transcript_ms > end_ms:
                    continue
                chapter_transcript += f"{transcript['text']}\n"
            if chapter_transcript != "":
                chapter.transcript = chapter_transcript

        video_transcript = "\n".join(
            [f"## {ct.title}\n-----\n{ct.transcript}" for ct in video_chapters if ct.transcript])
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
        with ThreadPoolExecutor(max_workers=len(video_chapters)) as executor:
            if provider == "gemini":
                futures = [executor.submit(VideoService.__analysis_video_with_gemini, chapter) for chapter in
                           video_chapters]
            elif provider == "local":
                futures = [executor.submit(VideoService.__analysis_video_with_local, chapter) for chapter in
                           video_chapters]
            elif provider == "mistral":
                futures = [executor.submit(VideoService.__analysis_video_with_mistral, chapter) for chapter in
                           video_chapters]
            elif provider == "openai":
                futures = [executor.submit(VideoService.__analysis_video_with_openai, chapter) for chapter in
                           video_chapters]
            elif provider == "voyageai":
                futures = [executor.submit(VideoService.__analysis_video_with_voyageai, chapter) for chapter in
                           video_chapters]
            else:
                logger.debug(f"selected provider: {provider}")
                raise AiError("unknown embedding provider")
        return sum(
            future.result() for future in concurrent.futures.as_completed(futures)
        )

    @staticmethod
    def __get_video_chapters(video: Video) -> list[VideoChapter]:
        video_chapters = list(
            VideoChapter.select().where(VideoChapter.video == video).order_by(VideoChapter.chapter_no))
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
    async def summary_video(vid: int, lang_code: str, provider: str, model: str = None) -> str:
        video: Video = VideoService.find_video_by_id(vid)
        if video is None:
            raise VideoError("video is not found")
        video.summary = VideoService.__summary_content(lang_code, model, provider, video)
        video.save()
        logger.debug("finish summary video")
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
        return AiService.chat_with_ai(provider=provider, model=model, question=prompt, system_prompt=SUMMARY_PROMPT)

    @staticmethod
    async def analysis_summary_video(vid: int, model: str, provider: str):
        video: Video = VideoService.find_video_by_id(vid)
        if video is None:
            raise VideoError("video is not found")
        if video.analysis_summary_state in [constants.ANALYSIS_STAGE_COMPLETED, constants.ANALYSIS_STAGE_PROCESSING]:
            return
        logger.debug("start analysis summary video")
        VideoService.__update_analysis_summary_state(video, constants.ANALYSIS_STAGE_PROCESSING)
        try:
            system_summary = VideoService.__summary_content(video.language, model, provider, video)
            texts, embeddings = AiService.get_texts_embedding(video.embedding_provider, system_summary)
            ids: list[str] = []
            documents: list[str] = []
            for index, text in enumerate(texts):
                ids.append(f"{video.id}_0_{index}")
                documents.append(f"## Summary - Part {index + 1}: \n---\n{text}")
            AiService.store_embeddings(f"video_summary_{video.id}", ids, texts, embeddings)
            video.analysis_summary_state = constants.ANALYSIS_STAGE_COMPLETED
            video.save()
            logger.debug("finish analysis summary video")
        except Exception as e:
            VideoService.__update_analysis_summary_state(video, constants.ANALYSIS_STAGE_INITIAL)
            logger.debug("fail to analysis summary video")
            raise e

    @staticmethod
    def delete(video_id: int):
        with sqlite_client.atomic() as transaction:
            try:
                video = VideoService.find_video_by_id(video_id)
                chapters = list(VideoChapter.select().where(VideoChapter.video == video))
                video.delete_instance()
                for chapter in chapters:
                    chapter.delete_instance()
                VideoService.delete_collection(f"video_{video_id}")
                transaction.commit()
            except Exception as e:
                transaction.rollback()
                raise e

    @staticmethod
    def delete_collection(table: str):
        chromadb_client.delete_collection(table)

    @staticmethod
    def get_videos_with_paging(page_no: int) -> tuple[int, list[Video]]:
        total = Video.select().count()
        limit = 48
        if total // limit < page_no:
            page_no = total // limit
        offset = (page_no - 1) * 48
        videos = list(Video.select().order_by(Video.id.desc()).offset(offset).limit(limit))
        video_data = [model_to_dict(video) for video in videos]
        return total, video_data
