FROM python:3.10-bookworm

RUN apt update && apt upgrade -y
RUN apt install g++ gcc cmake ffmpeg curl libssl-dev python3-pip -y
RUN pip install poetry
RUN apt autoremove -y
RUN apt autoclean -y

WORKDIR /app
COPY pyproject.toml pyproject.toml
RUN poetry install
COPY engine engine

EXPOSE 8000
CMD ["poetry", "python3", "engine/server.py"]