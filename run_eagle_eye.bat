@echo off
setlocal
cd /d "%~dp0"
title Eagle Eye AI — Surveillance & Security System

echo =====================================================================
echo               EAGLE EYE AI — INTELLIGENT SECURITY SURVEILLANCE
echo =====================================================================
echo.
echo Select an action:
echo  [1] Start Live Camera Surveillance (Laptop Webcam + Weapon Detection + Audio)
echo  [2] Launch Streamlit Security Dashboard
echo  [3] Start FastAPI Alert Backend
echo  [4] Run Security Audio & Siren Verification Test
echo  [5] Run Full Automated Test Suite
echo  [6] Exit
echo.
set /p opt="Enter choice [1-6]: "

if "%opt%"=="1" (
    echo.
    echo Starting Eagle Eye AI Live Surveillance Pipeline...
    echo Press 'q' in the camera window to stop.
    .venv\Scripts\python.exe -m src.main --config configs/default.yaml --profile eagle_eye_cpu
    pause
    goto end
)
if "%opt%"=="2" (
    echo.
    echo Launching Streamlit Security Dashboard...
    .venv\Scripts\python.exe -m streamlit run src/api/dashboard.py
    pause
    goto end
)
if "%opt%"=="3" (
    echo.
    echo Starting FastAPI Alert Ingestion & REST API on port 8000...
    .venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
    pause
    goto end
)
if "%opt%"=="4" (
    echo.
    echo Running Audio Subsystem Test (Beep + Emergency Siren)...
    .venv\Scripts\python.exe -m src.utils.security_audio --test
    pause
    goto end
)
if "%opt%"=="5" (
    echo.
    echo Running Full Test Suite...
    .venv\Scripts\python.exe -m pytest tests
    pause
    goto end
)

:end
endlocal
