FROM python:3.9

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 8001

WORKDIR /usr/src/ebl-ai
ENV VIRTUAL_ENV=/usr/src/ebl-ai/.venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
RUN pip3 install -U openmim
RUN mim install mmengine
RUN mim install "mmdet==3.0.0rc6"
RUN mim install mmocr


COPY ./ebl_ai ./ebl_ai
COPY ./model ./model
RUN cd model &&  gdown https://drive.google.com/uc?id=17vISNEucGkSyZSfWfu5eOrBtut2wXRJ6
CMD ["/bin/bash", "-c", "waitress-serve --port=8001 --call ebl_ai.app:get_app"]
