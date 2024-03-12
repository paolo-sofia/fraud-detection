FROM python:3.12-slim-bookworm
LABEL authors="paolo"

COPY ./data /data
COPY ./models/model /model

COPY .requirements.txt /app/requirements
RUN pip install -r /app/requirements.txt --no-cache-dir

WORKDIR /app

ENTRYPOINT [""]