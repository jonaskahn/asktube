import json

from peewee import Model, AutoField, CharField, TextField, IntegerField, BooleanField, ForeignKeyField

from backend.db.specs import sqlite_client


class Video(Model):
    id = AutoField(primary_key=True, index=True)
    youtube_id = CharField(index=True, unique=True)
    url = TextField(null=False)
    author = CharField(null=True)
    title = CharField(null=True)
    description = TextField(null=True)
    thumbnail = TextField(null=True)
    duration = IntegerField(null=False)
    amount_chapters = IntegerField(null=False)
    transcript = TextField(null=False, default="")
    summary = TextField(null=False, default="")
    is_analyzed = BooleanField(default=False)
    language = CharField(null=True)

    def __repr__(self):
        return f"Video(youtube_id={self.youtube_id}, url={self.url})"

    def set_languages(self, languages):
        self.languages = json.dumps(languages)

    def get_languages(self):
        return json.loads(self.languages.__str__())

    class Meta:
        database = sqlite_client


class VideoChapter(Model):
    id = AutoField(primary_key=True, index=True)
    video = ForeignKeyField(Video, backref="chapters", index=True)
    chapter_no = IntegerField(null=False)
    audio_path = TextField(null=True)
    title = CharField(null=False)
    transcript = TextField(null=False, default="")
    start_time = IntegerField(null=False, default=0)
    start_label = CharField(null=False, default="00:00:00")
    duration = IntegerField(null=False, default=0)

    def __repr__(self):
        return f"VideoChapter(title={self.title}, start_time={self.start_time}, start_label={self.start_label}, duration={self.duration})"

    class Meta:
        database = sqlite_client