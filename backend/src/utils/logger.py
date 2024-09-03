import logging

from sanic.log import logger

__sql_logger = logging.getLogger('peewee')
__sql_logger.addHandler(logging.StreamHandler())
__sql_logger.setLevel(logging.DEBUG)

log = logger
