from pytubefix import YouTube
from pytubefix.cli import on_progress


class YoutubeService:
    def __init__(self, url):
        self.__agent = YouTube(
            url=url,
            on_progress_callback=on_progress,
            use_oauth=False,
            allow_oauth_cache=True
        )

    def fetch_basic_info(self):
        captions = list(map(lambda c: {'name': c.name, 'value': c.code}, self.__agent.captions))
        return {
            'title': self.__agent.title,
            'description': self.__agent.description,
            'duration': self.__agent.length,
            'author': self.__agent.author,
            'thumbnail': self.__agent.thumbnail_url,
            'captions': captions
        }
