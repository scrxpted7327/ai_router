#!/bin/bash
# Push updates to the DigitalOcean droplet.
# Usage: ./deploy/deploy.sh root@<your-droplet-ip>
# First run: add --setup flag to also copy .env
set -euo pipefail

SERVER=${1:?"Usage: $0 <user@host> [--setup]"}
SETUP=${2:-}

if [[ "$SETUP" == "--setup" ]]; then
    echo "==> Copying .env to server..."
    scp .env "${SERVER}:/opt/ai-router/.env"
fi

echo "==> Deploying to ${SERVER}..."
ssh "$SERVER" bash <<'REMOTE'
set -euo pipefail
cd /opt/ai-router
git pull --ff-only
.venv/bin/pip install -r requirements.txt -q
systemctl restart ai-router
echo "Deploy complete."
systemctl status ai-router --no-pager -l
REMOTE
