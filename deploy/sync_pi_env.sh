#!/bin/bash
# Run on your PC after `pi login` — refreshes Google OAuth where possible and upserts pi keys into .env.
# Then: scp .env root@YOUR_DROPLET:/opt/ai-router/.env && ssh root@DROPLET 'chown airouter:airouter /opt/ai-router/.env && systemctl restart ai-router'
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python pi_auth.py "$@"
