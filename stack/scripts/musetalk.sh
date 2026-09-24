#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-up}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi
case "$command_name" in
  up)
    python3 scripts/preflight.py
    docker compose up -d musetalk kokoro kokoclone web
    docker compose ps musetalk kokoro kokoclone web
    ;;
  down)
    docker compose down
    ;;
  logs)
    docker compose logs -f --tail="${MUSE_TALK_LOG_LINES:-100}" musetalk kokoro kokoclone web
    ;;
  status)
    docker compose ps musetalk kokoro kokoclone web
    ;;
    *)
    printf 'usage: %s {up|down|logs|status}\n' "$0" >&2
    exit 2
    ;;
esac
