#!/bin/bash
# Run on the droplet after DNS (Cloudflare) points to this host and HTTP nginx works.
# Usage: sudo bash certbot-init.sh your.domain.com
# You will be prompted for an email and ToS unless certbot is already configured.
set -euo pipefail
DOMAIN="${1:?Usage: certbot-init.sh your.domain.com}"
apt-get update -y
apt-get install -y certbot python3-certbot-nginx
certbot --nginx -d "$DOMAIN"
systemctl reload nginx
echo "TLS OK. In Cloudflare: SSL/TLS = Full (strict). Set HTTPS=true in /opt/ai-router/.env"
