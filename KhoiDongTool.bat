@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Auto Tinh Kiem Truyen Ky
echo ========================================================
echo       DANG KHOI DONG AUTO TINH KIEM TRUYEN KY...
echo ========================================================

set PYTHON_CMD="C:\Users\nguye\AppData\Local\Programs\Python\Python311\python.exe"

if exist %PYTHON_CMD% (
    start "" %PYTHON_CMD% launcher.py
) else (
    start "" python launcher.py
)
exit
