FROM python:3.10-bookworm

SHELL [ "/bin/bash", "-c" ]

ENV SHELL=/bin/bash
ENV POETRY_HOME=/etc/poetry
ENV PATH="$POETRY_HOME/venv/bin:$PATH"
ENV AT_APP_DIR=/var/asktube

RUN apt update && apt upgrade -y
RUN apt install g++ gcc cmake ffmpeg curl libssl-dev bash -y
RUN apt install python3-pip pipx -y
RUN apt autoremove -y
RUN apt autoclean -y

#INSTALL POETRY
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 -

WORKDIR /usr/src/engine
COPY . .
RUN poetry env use $(which python3)
RUN poetry install

VOLUME [ "/var/asktube" ]
EXPOSE 8000
CMD ["poetry", "run", "python", "engine/server.py"]