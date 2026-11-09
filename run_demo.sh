#!/usr/bin/env bash
# Start the Leaffliction demo and open it in a browser.
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null || {
  echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
}
[ -d .venv ] || {
  echo "no .venv. run: uv venv --python 3.10 && uv pip install -r requirements.txt"
  exit 1
}
[ -f runs/cls/weights/best.pt ] || {
  echo "no weights. run: uv run python pipeline.py train"
  exit 1
}

PORT="${PORT:-8000}"
uv run uvicorn server:app --host 127.0.0.1 --port "$PORT" &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT INT TERM

for _ in $(seq 40); do
  curl -sf "http://127.0.0.1:$PORT/" -o /dev/null && break
  sleep 0.5
done
echo "demo at http://127.0.0.1:$PORT"
xdg-open "http://127.0.0.1:$PORT" 2>/dev/null || open "http://127.0.0.1:$PORT" 2>/dev/null || true
wait $PID
