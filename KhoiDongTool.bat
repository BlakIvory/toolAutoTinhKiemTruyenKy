@echo off
cd /d "%~dp0"

set PYTHONW_CMD="C:\Users\nguye\AppData\Local\Programs\Python\Python311\pythonw.exe"

if exist %PYTHONW_CMD% (
    start "" %PYTHONW_CMD% launcher.py
) else (
    start "" pythonw launcher.py 2>nul || start "" python launcher.py
)
exit
