FROM python:3.9

ENV PIPENV_VENV_IN_PROJECT 1

RUN pip install pipenv

EXPOSE 8001

WORKDIR /usr/src/ebl-ai

COPY Pipfile* ./
RUN pipenv install --dev

COPY ./ebl_ai ./ebl_ai


CMD ["pipenv", "run", "start"]
