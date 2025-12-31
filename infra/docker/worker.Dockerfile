FROM python:3.11-slim

WORKDIR /app
COPY libs ./libs
COPY services ./services
COPY prompts ./prompts

RUN pip install -U pip \
    && pip install -e libs/contracts -e libs/podcast -e services/worker

CMD ["python", "-m", "podcastify_worker.main"]
