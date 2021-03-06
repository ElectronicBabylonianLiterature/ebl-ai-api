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


COPY ./ebl_ai ./ebl_ai
COPY ./model ./model
RUN cd model &&  gdown https://drive.google.com/uc?id=1Z10LPzbxdCCrdxH_BruBJns5VkH3Pp7H
CMD ["/bin/bash", "-c", "waitress-serve --port=8001 --call ebl_ai.app:get_app"]
