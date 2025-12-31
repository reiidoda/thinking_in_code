# Thinking in Code - Enterprise TODO

A focused roadmap to move from a solid local-first prototype to an enterprise-ready platform.

## P0: Enterprise readiness
- [ ] Authentication and access control: API key rotation, optional JWT/SSO, role-based access (RBAC).
- [ ] Data governance: retention policies, scheduled purge jobs, and encryption-at-rest for artifacts.
- [ ] Security hardening: upload validation, size limits, MIME sniffing, and dependency vulnerability scans.
- [ ] Reliability: job idempotency, retry strategy with dead-letter queue, and worker concurrency controls.
- [ ] Observability: structured logs, trace propagation across API/worker, and alert-ready metrics.

## P1: Scale and integration
- [ ] Multi-tenant isolation with quotas, usage tracking, and per-tenant storage.
- [ ] Pluggable storage backends (S3/Blob) for artifacts and job metadata.
- [ ] Queue abstraction for Redis/Kafka with backpressure-aware scheduling.
- [ ] End-to-end integration tests with fixture PDFs and audio regression snapshots.
- [ ] SDK/CLI for job submission, progress streaming, and artifact download.

## P2: Product and UX
- [ ] Workspace/team management and audit logs per project.
- [ ] Voice catalog + pacing presets with per-organization defaults.
- [ ] Analytics dashboards (throughput, failure reasons, quality scores).
- [ ] Content review workflow (approve/reject segments before audio synthesis).

## Completed (initial build)
- [x] Restore missing services (API + worker) and wire them to the core pipeline.
- [x] Add Dockerfiles and align compose with actual service paths.
- [x] Dependency locks for core packages (API, worker, contracts, podcast).
- [x] Optional retrieval lock file (chromadb, sentence-transformers) generated with Python 3.11.
- [x] Validate the optional retrieval lock against Python 3.12 (no version drift vs 3.11).
- [x] CI with ruff/pytest across libs and services.
- [x] Pipeline + retrieval tests for GeneratePodcastFromPdf with mocked adapters.
- [x] Audio assembly tests for PydubAudioAssembler.
- [x] Persist job status/metrics with retry/backoff handling.
- [x] Ship .env examples for API/worker settings.
- [x] Create runtime folders automatically and document retention expectations.
- [x] Keep local-only artifacts out of VCS (.gitignore + docs).
- [x] Refresh README and add the web UI landing page.
- [x] Expose metrics/health endpoints for monitoring.
- [x] Collect user feedback and iterate on upload flow, job status clarity, and artifact browsing UX.
- [x] Extend narration personalization (voice profiles, pacing presets, per-segment emphasis).
