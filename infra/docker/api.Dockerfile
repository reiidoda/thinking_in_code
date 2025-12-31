FROM python:3.11-slim

WORKDIR /app
COPY libs ./libs
COPY services ./services
COPY prompts ./prompts
COPY docs ./docs
COPY Makefile ./

RUN pip install -U pip \
    && pip install -e libs/contracts -e libs/podcast -e services/api

EXPOSE 8000
CMD ["uvicorn", "podcastify_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
