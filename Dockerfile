FROM python:3.9

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV PIPENV_VENV_IN_PROJECT 1

RUN pip install pipenv

EXPOSE 8001

WORKDIR /usr/src/ebl-ai

COPY Pipfile* ./

RUN MMCV_WITH_OPS=1 pipenv install --dev --skip-lock


COPY ./ebl_ai ./ebl_ai
COPY ./model ./model


CMD ["pipenv", "run", "start"]
