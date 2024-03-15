FROM python:3.12-slim-bookworm
LABEL authors="paolo"

ENV PATH /root/.local/bin:${PATH}

COPY ./models/ /models
COPY ./data/columns /data/columns

RUN apt update && apt install -y --no-install-recommends apt-utils curl libgomp1

COPY src/fraud_detection/requirements.txt .
RUN pip install -r requirements.txt --use-pep517 --no-cache-dir --user \
    && find /root/.local/ -follow -type f  \
    -name '*.a' -name '*.txt' -name '*.md' -name '*.png' \
    -name '*.jpg' -name '*.jpeg' -name '*.js.map' -name '*.pyc' \
    -name '*.c' -name '*.pxc' -name '*.pyd' \
    -delete \
    && find /usr/local/lib/python3.12/ -name '__pycache__' | xargs rm -r \
    && rm -r /var/cache/debconf/templates* \
    && rm -r /var/lib/dpkg/status* \
    && rm -rf /var/log/

COPY src/ /app/src

WORKDIR /app

CMD ["uvicorn", "src.fraud_detection.inference.main:app", "--host", "0.0.0.0", "--port", "8000"]