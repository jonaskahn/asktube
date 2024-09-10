import asyncio
from json import dumps

from playhouse.shortcuts import model_to_dict
from sanic import Sanic, Request
from sanic import json, text

from engine.assistants.errors import LogicError
from engine.assistants.logger import log
from engine.database.models import Video, VideoChapter, Chat
from engine.database.specs import sqlite_client
from engine.services.video_service import VideoService
from engine.services.youtube_service import YoutubeService

Sanic.START_METHOD_SET = True
Sanic.start_method = "fork"
app = Sanic("AskTube", dumps=dumps)

app.config.KEEP_ALIVE = False
app.config.REQUEST_TIMEOUT = 90
app.config.RESPONSE_TIMEOUT = 300


@app.listener('before_server_start')
async def connect_db(app, loop):
    log.debug("open sqlite Connection")
    sqlite_client.connect()
    sqlite_client.create_tables([Video, VideoChapter, Chat])


@app.listener('after_server_stop')
async def close_db(app, loop):
    if not sqlite_client.is_closed():
        log.debug("close sqlite")
        sqlite_client.close()


@app.exception(Exception)
async def handle_exception(request: Request, exception: Exception):
    log.error(exception, exc_info=True)
    return json(
        {
            "error": {
                "message": str(exception),
                "status_code": 500
            }
        },
        status=500
    )


@app.exception(LogicError)
async def handle_exception(request: Request, exception: LogicError):
    log.error(exception, exc_info=True)
    return json(
        {
            "error": {
                "message": str(exception),
                "status_code": 400
            }
        },
        status=500
    )


@app.get("/api/health")
async def health(request: Request):
    return text("API is running!!!")


@app.post("/api/youtube/prefetch")
async def fetch_youtube_video_info(request: Request):
    url = request.json['url']
    youtube_service = YoutubeService(url)
    data = youtube_service.fetch_basic_info()
    return json({
        "message": "Successfully fetch video info",
        "payload": data
    })


@app.post("/api/youtube/process")
async def process_youtube_video(request: Request):
    url = request.json['url']
    provider = request.json['provider']
    youtube_service = YoutubeService(url)
    video = await asyncio.create_task(youtube_service.fetch_video_data())
    asyncio.create_task(VideoService.analysis_video(video.id, provider))
    return json({
        "message": "Successfully fetch video data, analyze video in processing.",
        "payload": model_to_dict(video)
    })


@app.post("/api/youtube/analysis")
async def analysis_youtube_video(request: Request):
    video_id = request.json['video_id']
    provider = request.json['provider']
    asyncio.create_task(VideoService.analysis_video(video_id, provider))
    return json({
        "message": "Analyze video in processing.",
    })


@app.post("/api/video/chat")
async def chat(request: Request):
    vid = request.json['video_id']
    question = request.json['question']
    provider = request.json['provider']
    model = request.json.get("model", None)
    data = await  asyncio.create_task(VideoService.ask(question, vid, provider, model))
    return json({
        "message": "Successfully",
        "payload": data
    })


@app.post("/api/video/summary")
async def summary(request: Request):
    vid = request.json['video_id']
    lang_code = request.json['lang_code']
    provider = request.json['provider']
    model = request.json.get("model", None)
    data = await asyncio.create_task(VideoService.summary_video(vid, lang_code, provider, model))
    return json({
        "message": "Successfully summary video",
        "payload": data
    })


@app.delete("/api/video/<video_id>")
async def delete_video(request: Request, video_id: int):
    VideoService.delete(video_id)
    return json({
        "message": f"Successfully delete video {video_id}"
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, access_log=True, debug=True)
