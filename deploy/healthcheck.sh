#!/bin/bash
# Optional cron or systemd timer: alert if app or disk is unhealthy.
set -euo pipefail
URL="${HEALTH_URL:-http://127.0.0.1:4000/health}"
if ! curl -sf "$URL" | grep -q '"status"'; then
  echo "healthcheck FAILED: $URL" >&2
  exit 1
fi
# Warn if root filesystem > 90% full
USE=$(df / | awk 'NR==2 {gsub(/%/,"",$5); print $5}')
if [[ "${USE:-0}" -gt 90 ]]; then
  echo "disk warning: ${USE}% used on /" >&2
fi
exit 0
