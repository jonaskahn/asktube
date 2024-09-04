import asyncio
from json import dumps

from sanic import Sanic, json, text

from backend.db.models import Video, VideoChapter
from backend.db.specs import sqlite_client
from backend.error.base import LogicError
from backend.services.youtube_service import YoutubeService
from backend.utils.logger import log

app = Sanic("AskTube", dumps=dumps)
app.config.KEEP_ALIVE = False
app.config.REQUEST_TIMEOUT = 120
app.config.RESPONSE_TIMEOUT = 600


@app.listener('before_server_start')
async def connect_db(app, loop):
    log.debug("Open DB Connection")
    sqlite_client.connect()
    sqlite_client.create_tables([Video, VideoChapter])


@app.listener('after_server_stop')
async def close_db(app, loop):
    if not sqlite_client.is_closed():
        log.debug("Closing DB")
        sqlite_client.close()


@app.exception(Exception)
async def handle_exception(request, exception: Exception):
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
async def handle_exception(request, exception: LogicError):
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
async def health(request):
    return text("API is running!!!")


@app.post("/api/youtube/prefetch")
async def fetch_youtube_video_info(request):
    url = request.json['url']
    youtube_service = YoutubeService(url)
    data = youtube_service.fetch_basic_info()
    return json({
        "message": "Successfully fetched video info",
        "payload": data
    })


@app.post("/api/youtube/process")
async def process_youtube_video(request):
    url = request.json['url']
    youtube_service = YoutubeService(url)
    await asyncio.create_task(youtube_service.fetch_video_data())
    return json({
        "message": "Successfully fetched video data",
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, access_log=True, debug=True)
