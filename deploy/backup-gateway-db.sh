#!/bin/bash
# Cron: 0 3 * * * /opt/ai-router/deploy/backup-gateway-db.sh
set -euo pipefail
APP_DIR="${APP_DIR:-/opt/ai-router}"
DB="$APP_DIR/data/gateway.db"
DEST="${BACKUP_DIR:-$APP_DIR/data/backups}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$DEST"
if [[ -f "$DB" ]]; then
  sqlite3 "$DB" ".backup '$DEST/gateway-$STAMP.db'"
  find "$DEST" -name 'gateway-*.db' -mtime +14 -delete 2>/dev/null || true
  echo "Backed up to $DEST/gateway-$STAMP.db"
else
  echo "No database at $DB"
fi
