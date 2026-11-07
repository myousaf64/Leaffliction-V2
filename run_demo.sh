#!/usr/bin/env bash
# Start the Leaffliction demo and open it in a browser.
set -euo pipefail
cd "$(dirname "$0")"

[ -d .venv ] || { echo "no .venv. run: python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
[ -f runs/cls/weights/best.pt ] || { echo "no weights. run: .venv/bin/python pipeline.py train"; exit 1; }

PORT="${PORT:-8000}"
.venv/bin/uvicorn server:app --host 127.0.0.1 --port "$PORT" &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT INT TERM

for _ in $(seq 40); do
  curl -sf "http://127.0.0.1:$PORT/" -o /dev/null && break
  sleep 0.5
done
echo "demo at http://127.0.0.1:$PORT"
xdg-open "http://127.0.0.1:$PORT" 2>/dev/null || open "http://127.0.0.1:$PORT" 2>/dev/null || true
wait $PID
