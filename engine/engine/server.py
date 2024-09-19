import asyncio
import platform
from json import dumps

from sanic import Sanic, Request, response
from sanic import json, text
from sanic.log import logger
from sanic.worker.manager import WorkerManager

from engine.cors import add_cors_headers
from engine.database.models import Video, VideoChapter, Chat
from engine.database.specs import sqlite_client
from engine.services.chat_service import ChatService
from engine.services.video_service import VideoService
from engine.services.youtube_service import YoutubeService
from engine.supports.errors import LogicError
from engine.supports.logger import setup_log

app = Sanic("AskTube", dumps=dumps)
app.register_middleware(add_cors_headers, "response")
app.config.KEEP_ALIVE = False

WorkerManager.THRESHOLD = 600
if platform.system() == "Linux":
    logger.debug('use "fork" for multi-process')
    Sanic.start_method = "fork"
else:
    logger.debug('use "spawn" for multi-process')


@app.listener('before_server_start')
async def connect_db(app, loop):
    logger.debug("open sqlite Connection")
    sqlite_client.connect()
    sqlite_client.create_tables([Video, VideoChapter, Chat])


@app.listener('after_server_stop')
async def close_db(app, loop):
    if not sqlite_client.is_closed():
        logger.debug("close sqlite")
        sqlite_client.close()


@app.exception(Exception)
async def handle_exception(request: Request, exception: Exception):
    logger.error(exception, exc_info=True)
    return json(
        {
            "status_code": 500,
            "message": str(exception)
        },
        status=500
    )


@app.exception(LogicError)
async def handle_exception(request: Request, exception: LogicError):
    logger.error(exception, exc_info=True)
    return json(
        {
            "status_code": 400,
            "message": str(exception),
        },
        status=500
    )


@app.get("/api/health")
async def health(request: Request):
    return text("API is running!!!")


@app.options("/api/youtube/prefetch")
async def opts_fetch_youtube_video_info(request: Request):
    return response.HTTPResponse(status=200)


@app.post("/api/youtube/prefetch")
async def fetch_youtube_video_info(request: Request):
    url = request.json['url']
    youtube_service = YoutubeService(url)
    data = youtube_service.fetch_basic_info()
    return json({
        "status_code": 200,
        "message": "Successfully fetch video info",
        "payload": data
    })


@app.options("/api/youtube/process")
async def opts_process_youtube_video(request: Request):
    return response.HTTPResponse(status=200)


@app.post("/api/youtube/process")
async def process_youtube_video(request: Request):
    url = request.json['url']
    provider = request.json['provider']
    youtube_service = YoutubeService(url)
    data = await asyncio.create_task(youtube_service.fetch_video_data(provider))
    return json({
        "status_code": 200,
        "message": "Successfully fetch video data, analyze video in processing.",
        "payload": data
    })


@app.options("/api/video/analysis")
async def opts_analysis_youtube_video(request: Request):
    return response.HTTPResponse(status=200)


@app.post("/api/video/analysis")
async def analysis_youtube_video(request: Request):
    video_id: int = int(request.json['video_id'])
    data = await asyncio.create_task(VideoService.analysis_video(video_id))
    return json({
        "status_code": 200,
        "message": "Analyze video in processing.",
        "payload": data
    })


@app.options("/api/video/detail/<video_id>")
async def opts_detail(request: Request, video_id: int):
    return response.HTTPResponse(status=200)


@app.get("/api/video/detail/<video_id>")
async def get_video_detail(request: Request, video_id: int):
    data = VideoService.get_video_detail(video_id)
    return json({
        "status_code": 200,
        "message": "Successfully get video detail",
        "payload": data
    })


@app.options("/api/video/summary")
async def opts_summary(request: Request):
    return response.HTTPResponse(status=200)


@app.post("/api/video/summary")
async def summary(request: Request):
    vid = request.json['video_id']
    lang_code = request.json['lang_code']
    provider = request.json['provider']
    model = request.json.get("model", None)
    data = await asyncio.create_task(VideoService.summary_video(vid, lang_code, provider, model))
    asyncio.create_task(VideoService.analysis_summary_video(vid, model, provider))
    return json({
        "status_code": 200,
        "message": "Successfully summary video",
        "payload": data
    })


@app.options("/api/video/<video_id>")
async def opts_list_video(request: Request):
    return response.HTTPResponse(status=200)


@app.delete("/api/video/<video_id>")
async def delete_video(request: Request, video_id: int):
    VideoService.delete(video_id)
    return json({
        "status_code": 200,
        "message": f"Successfully delete video {video_id}"
    })


@app.options("/api/videos/<page>")
async def opts_list_videos(request: Request, page: int):
    return response.HTTPResponse(status=200)


@app.get("/api/videos/<page>")
async def list_videos(request: Request, page: int):
    total, data = VideoService.get_videos_with_paging(page)
    return json({
        "status_code": 200,
        "message": "Successfully",
        "payload": {
            "total": total,
            "videos": data
        }
    })


@app.options("/api/chat")
async def opts_chat(request: Request):
    return response.HTTPResponse(status=200)


@app.post("/api/chat")
async def chat(request: Request):
    vid = request.json['video_id']
    question = request.json['question']
    provider = request.json['provider']
    model = request.json.get("model", None)
    data = await asyncio.create_task(ChatService.ask(question, vid, provider, model))
    return json({
        "status_code": 200,
        "message": "Successfully",
        "payload": data
    })


@app.options("/api/chat/history/<video_id>")
async def opts_chat_history(request: Request, video_id: int):
    return response.HTTPResponse(status=200)


@app.get("/api/chat/history/<video_id>")
async def chat_history(request: Request, video_id: int):
    chat_histories = ChatService.get_chat_histories(video_id=video_id)
    return json({
        "status_code": 200,
        "message": "Successfully",
        "payload": chat_histories
    })


@app.options("/api/chat/clear/<video_id>")
async def opts_clear_chat(request: Request, video_id: int):
    return response.HTTPResponse(status=200)


@app.delete("/api/chat/clear/<video_id>")
async def clear_chat(request: Request, video_id: int):
    ChatService.clear_chat(video_id)
    return response.HTTPResponse(status=200)


if __name__ == '__main__':
    setup_log()
    app.run(host="0.0.0.0", port=8000, access_log=True, dev=True, debug=True, workers=10)
