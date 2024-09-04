class VideoError(Exception):
    pass


class VideoNotFoundError(VideoError):
    pass


class VideoNotAnalyzedError(VideoError):
    pass
