@echo off
title SignalBoard
cd /d "%~dp0"
echo Starting SignalBoard...
start "" http://localhost:8000
python main.py serve --config config.yaml
pause
