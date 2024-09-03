from json import dumps

from sanic import Sanic, json, text

from backend.src.db.models import Video, VideoChapter
from backend.src.db.specs import sqlite_client
from backend.src.error.base import LogicException
from backend.src.services.youtube_service import YoutubeService
from backend.src.utils.logger import log

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


@app.exception(LogicException)
async def handle_exception(request, exception: LogicException):
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
async def fetch_video_info(request):
    url = request.json['url']
    youtube_service = YoutubeService(url)
    data = youtube_service.fetch_basic_info()
    return json({
        "message": "Successfully fetched video info",
        "payload": data
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, access_log=True, debug=True)
