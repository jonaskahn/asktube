import logging

from sanic.log import logger

from backend.assistants.env import DEBUG_MODE

if DEBUG_MODE == "on":
    logging.getLogger('peewee').addHandler(logging.StreamHandler())
    logging.getLogger('peewee').setLevel(logging.DEBUG)

    logging.getLogger('faster_whisper').addHandler(logging.StreamHandler())
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

log = logger
