@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" server.py --warmup %*
endlocal
