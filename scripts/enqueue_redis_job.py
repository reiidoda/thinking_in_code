#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import uuid

from podcastify_worker import redis_queue


def main() -> None:
    parser = argparse.ArgumentParser(description="Enqueue a podcast job into Redis sorted-set queue.")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument("--key", default="podcastify:jobs", help="Redis sorted-set key")
    parser.add_argument("--job-id", help="Job ID (default: generated)")
    parser.add_argument("--priority", type=float, default=None, help="Override priority (higher is earlier)")
    parser.add_argument("--num-pdfs", type=int, default=None, help="Hint: number of PDFs (for priority scoring)")
    parser.add_argument("--minutes", type=int, default=None, help="Hint: target minutes (for priority scoring)")
    parser.add_argument("--meta", type=str, default=None, help="Optional JSON metadata to store alongside job")
    args = parser.parse_args()

    job_id = args.job_id or f"job-{uuid.uuid4().hex[:8]}"
    meta = json.loads(args.meta) if args.meta else {}
    score = redis_queue.enqueue(
        args.redis_url,
        args.key,
        job_id=job_id,
        priority=args.priority,
        meta={"num_pdfs": args.num_pdfs, "minutes": args.minutes, **meta},
    )
    print(f"Enqueued job_id={job_id} priority={score:.3f} redis={args.redis_url} key={args.key}")


if __name__ == "__main__":
    main()
