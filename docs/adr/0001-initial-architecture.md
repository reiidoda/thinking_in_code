# ADR-0001: Thinking in Code monorepo architecture

Date: 2025-12-31

## Decision
Use a monorepo with:
- deployable services (`services/api`, `services/worker`)
- shared libs (`libs/contracts`, `libs/podcast`)
- infrastructure (`infra/docker`)

## Rationale
- scales with features (RAG, TTS, audio) without turning API into a “god service”
- enables independent deployment and scaling of worker
- enforces clean boundaries via shared contracts

## Consequences
- slightly more setup (multiple pyprojects)
- clearer ownership and long-term maintainability
