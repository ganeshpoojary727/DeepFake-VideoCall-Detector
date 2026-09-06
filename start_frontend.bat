@echo off
title DeepFake Detector - Next.js Frontend
echo ============================================================
echo   Starting DeepFake Detector - Next.js Frontend (Port 3000)
echo ============================================================
echo.

cd frontend
if not exist node_modules (
    echo [INFO] node_modules not found. Installing frontend dependencies...
    call npm install
)

call npm run dev
pause
