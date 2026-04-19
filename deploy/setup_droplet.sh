#!/bin/bash
# Run once as root on a fresh Ubuntu 24.04 LTS Droplet.
# Usage: ssh root@<your-ip> 'bash -s' < deploy/setup_droplet.sh
set -euo pipefail

echo "==> Updating system..."
apt-get update -y && apt-get upgrade -y

echo "==> Installing system deps..."
apt-get install -y git nginx curl ufw software-properties-common

echo "==> Installing Python 3.13..."
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
apt-get install -y python3.13 python3.13-venv python3.13-dev

echo "==> Creating service user..."
id -u airouter &>/dev/null || useradd -m -s /bin/bash airouter

echo "==> Creating app directory..."
mkdir -p /opt/ai-router
chown airouter:airouter /opt/ai-router

echo "==> Configuring firewall..."
ufw allow OpenSSH
ufw allow 'Nginx HTTP'
ufw --force enable

echo ""
echo "Done. Next steps:"
echo "  1. su - airouter"
echo "  2. git clone https://github.com/<your-user>/ai_router /opt/ai-router"
echo "  3. cd /opt/ai-router"
echo "  4. python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt"
echo "  5. Copy .env:  scp .env root@<ip>:/opt/ai-router/.env"
echo "  6. Install service + nginx:"
echo "       cp deploy/ai-router.service /etc/systemd/system/"
echo "       cp deploy/nginx-ai.conf /etc/nginx/sites-available/ai-router"
echo "       ln -s /etc/nginx/sites-available/ai-router /etc/nginx/sites-enabled/"
echo "       systemctl daemon-reload"
echo "       systemctl enable --now ai-router"
echo "       nginx -t && systemctl reload nginx"
