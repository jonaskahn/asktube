import logging

from sanic.log import logger

from engine.supports import env


def setup_log():
    if env.DEBUG_MODE in ["on", "yes", "enabled"]:
        logging.getLogger('peewee').addHandler(logging.StreamHandler())
        logging.getLogger('peewee').setLevel(logging.DEBUG)
        logging.getLogger('faster_whisper').addHandler(logging.StreamHandler())
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)


setup_log()
log = logger
