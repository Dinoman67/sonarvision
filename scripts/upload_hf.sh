#!/usr/bin/env bash
set -e

# Usage:
#   ./scripts/upload_hf.sh [<HF_TOKEN>] [<HF_USERNAME>]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
HF_BIN="$ROOT_DIR/.venv/bin/hf"

if [ ! -f "$HF_BIN" ]; then
  echo "❌ Error: hf CLI not found at $HF_BIN"
  exit 1
fi

HF_TOKEN="${1:-$HF_TOKEN}"
HF_USER="${2:-}"

# If a token was supplied explicitly, log in with it
if [ -n "$HF_TOKEN" ]; then
  echo "🔑 Authenticating with provided token..."
  "$HF_BIN" auth login --token "$HF_TOKEN"
fi

# Detect authenticated username
CURRENT_USER=$("$HF_BIN" auth whoami --format json 2>/dev/null | grep -o '"user": *"[^"]*"' | cut -d'"' -f4 || true)

if [ -z "$CURRENT_USER" ] && [ -z "$HF_USER" ]; then
  echo "❌ Error: Not authenticated and no token provided."
  echo "Usage: ./scripts/upload_hf.sh <YOUR_HF_TOKEN>"
  exit 1
fi

HF_USER="${HF_USER:-$CURRENT_USER}"

echo "👤 Authenticated User: $HF_USER"
echo "🔒 Visibility: Private / Restricted"

MODEL_REPO="$HF_USER/sonarvision-yolov8-esi-v6"
DATASET_REPO="$HF_USER/sonarvision-multisource-v6"

echo "============================================================"
echo "📦 [1/2] Uploading Model to: https://huggingface.co/$MODEL_REPO"
echo "============================================================"
"$HF_BIN" upload "$MODEL_REPO" "$ROOT_DIR/hf_export_v6/model" . --repo-type model --private

echo ""
echo "============================================================"
echo "🗂️  [2/2] Uploading Dataset (924 MB) to: https://huggingface.co/datasets/$DATASET_REPO"
echo "============================================================"
"$HF_BIN" upload "$DATASET_REPO" "$ROOT_DIR/hf_export_v6/dataset" . --repo-type dataset --private

echo ""
echo "============================================================"
echo "🎉 Done! Both repositories are live (Restricted/Private):"
echo "• Model:   https://huggingface.co/$MODEL_REPO"
echo "• Dataset: https://huggingface.co/datasets/$DATASET_REPO"
echo "============================================================"
