from pathlib import Path

import chromadb
from peewee import SqliteDatabase

from engine.supports import env, constants

Path(env.APP_DIR).mkdir(parents=True, exist_ok=True)

sqlite_client = SqliteDatabase(database=constants.SQL_DATABASE)

chromadb_client = chromadb.PersistentClient(path=constants.VECTOR_DATABASE)
