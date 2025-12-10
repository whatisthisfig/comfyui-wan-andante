#!/usr/bin/env bash
set -euo pipefail

# Use the start.sh provided by this image instead of pulling an external repo
chmod +x /start.sh
exec /start.sh