@echo off
title DeepFake Detector - FastAPI Backend
echo ============================================================
echo   Starting DeepFake Detector - FastAPI Backend (Port 8000)
echo ============================================================
echo.

if not exist .venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment (.venv) not found!
    echo Please run: python -m venv .venv and install dependencies first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python -m uvicorn app.api.server:create_app --factory --host 127.0.0.1 --port 8000
pause
