#!/usr/bin/env bash
set -euo pipefail

echo "Health:"
curl -s http://localhost:8000/health | python -m json.tool
