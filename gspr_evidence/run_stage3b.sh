#!/usr/bin/env bash
set -euo pipefail
# Manual entrypoint only: read Stage-3A before starting this script.
STAGE=3B
: "${STAGE3A_ROOT:?Read Stage-3A first, then supply its completed output root}"
source "$(dirname "${BASH_SOURCE[0]}")/stage3_launch.sh"
