#!/usr/bin/env bash
set -e

# SonarVision Unified Application Launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
FRONTEND_DIST="$SCRIPT_DIR/frontend/dist"

usage() {
    echo "Usage: ./start.sh [options]"
    echo ""
    echo "Options:"
    echo "  --dev            Run backend and frontend dev server concurrently (hot reloading)"
    echo "  --build          Force rebuild frontend assets before starting"
    echo "  --host HOST      Host address to bind (default: 0.0.0.0)"
    echo "  --port PORT      Port number to bind (default: 8000)"
    echo "  --help           Show this help message"
    echo ""
}

DEV_MODE=false
FORCE_BUILD=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --build)
            FORCE_BUILD=true
            shift
            ;;
        --host)
            HOST="$2"
            EXTRA_ARGS+=("--host" "$2")
            shift 2
            ;;
        --port)
            PORT="$2"
            EXTRA_ARGS+=("--port" "$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Ensure frontend build exists for unified serving
if [ "$DEV_MODE" = false ]; then
    if [ ! -d "$FRONTEND_DIST" ] || [ "$FORCE_BUILD" = true ]; then
        echo "[1/2] Building frontend production bundle..."
        (cd frontend && npm install && npm run build)
    else
        echo "[1/2] Frontend build verified."
    fi
    echo "[2/2] Launching unified SonarVision web app on http://${HOST}:${PORT}..."
    exec python3 run_app.py --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}"
else
    echo "[Dev Mode] Starting backend (port $PORT) and frontend dev server (port 5173)..."
    
    cleanup() {
        echo ""
        echo "Shutting down servers..."
        kill 0
    }
    trap cleanup SIGINT SIGTERM EXIT

    python3 run_app.py --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}" &
    BACKEND_PID=$!

    (cd frontend && npm run dev) &
    FRONTEND_PID=$!

    wait
fi
