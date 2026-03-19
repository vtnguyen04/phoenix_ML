#!/bin/bash
# Entrypoint script: fix volume permissions then run as phoenix user
set -e

# Fix ownership on mounted volumes (runs as root)
chown -R phoenix:phoenix /app/models /app/data 2>/dev/null || true

# Drop to phoenix user and exec the CMD
exec gosu phoenix "$@"
