FROM python:3.9

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 8001

WORKDIR /usr/src/ebl-ai
RUN python -m venv /usr/src/ebl-ai/.venv
COPY requirements.txt ./
RUN . /usr/src/ebl-ai/.venv/bin/activate  && pip3 install -r requirements.txt


COPY ./ebl_ai ./ebl_ai
COPY ./model ./model


CMD [". /usr/src/ebl-ai/.venv/bin/activate && exec waitress-serve --port=8001 --call ebl.app:get_app"]
