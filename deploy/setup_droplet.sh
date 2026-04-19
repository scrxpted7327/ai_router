#!/bin/bash
# Full bootstrap — paste into DigitalOcean startup script field.
# After boot, only one step remains: scp your .env to /opt/ai-router/.env
set -euo pipefail

REPO="https://github.com/scrxpted7327/ai_router.git"
APP_DIR="/opt/ai-router"
SERVICE_USER="airouter"

echo "==> Updating system..."
apt-get update -y && apt-get upgrade -y

echo "==> Installing system deps..."
apt-get install -y git nginx curl ufw software-properties-common

echo "==> Installing Python 3.13..."
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
apt-get install -y python3.13 python3.13-venv python3.13-dev

echo "==> Creating service user..."
id -u $SERVICE_USER &>/dev/null || useradd -m -s /bin/bash $SERVICE_USER

echo "==> Cloning repo..."
git clone "$REPO" "$APP_DIR"
chown -R $SERVICE_USER:$SERVICE_USER "$APP_DIR"

echo "==> Installing Python deps..."
sudo -u $SERVICE_USER python3.13 -m venv "$APP_DIR/.venv"
sudo -u $SERVICE_USER "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

echo "==> Installing systemd service..."
cp "$APP_DIR/deploy/ai-router.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable ai-router

echo "==> Installing nginx config..."
cp "$APP_DIR/deploy/nginx-ai.conf" /etc/nginx/sites-available/ai-router
ln -sf /etc/nginx/sites-available/ai-router /etc/nginx/sites-enabled/ai-router
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl enable nginx && systemctl reload nginx

echo "==> Configuring firewall..."
ufw allow OpenSSH
ufw allow 'Nginx HTTP'
ufw --force enable

echo ""
echo "======================================"
echo "Bootstrap complete."
echo "ONE STEP REMAINING:"
echo "  scp .env root@\$(hostname -I | awk '{print \$1}'):/opt/ai-router/.env"
echo "  systemctl start ai-router"
echo "======================================"
