@echo off
REM QuantForge launcher - always uses the venv Python
REM Usage: quantforge.bat <script_args>
REM Example: quantforge.bat -m quantforge.scripts.run_baseline --max_samples 64

SET SCRIPT_DIR=%~dp0
SET VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe

IF NOT EXIST "%VENV_PYTHON%" (
    echo ERROR: venv not found at %VENV_PYTHON%
    echo Run: python -m venv .venv
    echo Then: .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

"%VENV_PYTHON%" %*
