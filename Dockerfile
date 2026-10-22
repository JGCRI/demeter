FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# a Java runtime is required by gcamreader to query GCAM databases
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY . /code

RUN pip install --no-cache-dir ".[gcam]"

ENTRYPOINT ["python", "-m", "demeter.model"]
