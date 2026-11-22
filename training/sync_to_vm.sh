#!/usr/bin/env bash
# Copy only what the VM needs: the training scripts, the built TimeNet registry, OpenTSLM and the
# timenet package. Run from anywhere:  ./training/sync_to_vm.sh <user> <ip>
set -euo pipefail
USER_NAME="${1:?usage: sync_to_vm.sh <user> <ip>}"
HOST="${2:?usage: sync_to_vm.sh <user> <ip>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ssh -o BatchMode=yes "$USER_NAME@$HOST" "mkdir -p ~/forecasttrust/TimeNet/packages"
rsync -az --delete --exclude '__pycache__' --exclude '.pytest_cache' \
  "$ROOT/training" "$ROOT/registry" "$ROOT/OpenTSLM" "$USER_NAME@$HOST:~/forecasttrust/"
rsync -azL --delete --exclude "__pycache__" --exclude ".venv" \
  "$ROOT/TimeNet/packages/timenet" "$USER_NAME@$HOST:~/forecasttrust/TimeNet/packages/"
echo "synced to $USER_NAME@$HOST:~/forecasttrust"
