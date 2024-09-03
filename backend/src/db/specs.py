from pathlib import Path

import chromadb
from peewee import SqliteDatabase

from backend.src.constants import SQL_DATABASE, VECTOR_DATABASE
from backend.src.env import APP_DIR

Path(APP_DIR).mkdir(parents=True, exist_ok=True)

sqlite_client = SqliteDatabase(database=SQL_DATABASE)

chromadb_client = chromadb.PersistentClient(path=VECTOR_DATABASE)
