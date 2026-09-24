#!/usr/bin/env bash
# Compatibility entrypoint retained from the archived patch. The repository-level
# script owns Compose lifecycle and all host paths now come from .env.
set -euo pipefail
case "${1:-up}" in
  stop|down) exec docker compose down ;;
  *) exec ./scripts/musetalk.sh "$@" ;;
esac
