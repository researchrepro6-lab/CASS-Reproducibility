@echo off
REM Windows launcher for the CASS reproducibility package.
REM   run_all.bat --check    check Python, packages and data
REM   run_all.bat --quick    smoke test of every script (temporary copy, shipped results untouched)
REM   run_all.bat            full run (about 1.5-2 h)
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 run_all.py %*) else (python run_all.py %*)
pause
