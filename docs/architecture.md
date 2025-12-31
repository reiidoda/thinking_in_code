# Thinking in Code Architecture

## Services
- API: request validation, job lifecycle, status endpoints.
- Worker: executes long-running pipeline jobs.

## Libraries
- contracts: shared schemas for job requests/results/events.
- podcast: clean-architecture implementation of the pipeline.

## Data flow (initial)
1. API receives PDF upload.
2. API creates a Job (writes `job.json` + stores PDF) and returns `job_id`.
3. Worker polls local queue (filesystem) and processes jobs.
4. Worker writes outputs (script/audio/transcript/metadata).
5. API serves job status and download URLs (later).

## Why contracts?
Contracts keep API and Worker stable as you refactor internals.
