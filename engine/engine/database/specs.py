import chromadb
from chromadb import Settings
from peewee import SqliteDatabase

from engine.supports import constants

sqlite_client = SqliteDatabase(database=constants.SQL_DATABASE)

chromadb_client = chromadb.PersistentClient(path=constants.VECTOR_DATABASE, settings=Settings(anonymized_telemetry=False))
