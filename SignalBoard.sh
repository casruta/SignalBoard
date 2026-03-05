#!/bin/bash
cd "$(dirname "$0")"
echo "Starting SignalBoard..."
python main.py serve --config config.yaml &
sleep 2
xdg-open http://localhost:8000 2>/dev/null || open http://localhost:8000 2>/dev/null
wait
