FROM ubuntu:22.04

RUN apt update && apt upgrade -y
RUN apt install build-essential cmake ffmpeg curl software-properties-common python3-pip -y
RUN pip install poetry
RUN apt autoremove -y
RUN apt autoclean -y

WORKDIR /app
COPY pyproject.toml pyproject.toml
RUN poetry install
COPY engine engine

EXPOSE 8000
CMD ["poetry", "python", "engine/server.py"]