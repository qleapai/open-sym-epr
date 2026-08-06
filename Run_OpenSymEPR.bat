@echo off
REM ============================================================
REM  Open-Sym-EPR  -  one-click local launcher (Windows)
REM  Opens the app in your browser at http://localhost:8501
REM ============================================================
cd /d "%~dp0"

REM -- If the environment is already built, just launch it --
if exist ".venv\Scripts\streamlit.exe" goto run

echo First run: creating a local Python environment and installing Open-Sym-EPR.
echo This happens only once and takes a couple of minutes...

REM -- Pick a compatible interpreter (numpy<2.0 needs Python 3.10-3.12) --
set "PYEXE="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYEXE ( py -%%V -c "import sys" >nul 2>&1 && set "PYEXE=py -%%V" )
)
if not defined PYEXE (
    where python >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
    echo.
    echo ERROR: No suitable Python found. Please install Python 3.11 from
    echo https://www.python.org/downloads/ and run this file again.
    pause
    exit /b 1
)

%PYEXE% -m venv .venv
if errorlevel 1 ( echo Failed to create the environment. & pause & exit /b 1 )
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
call ".venv\Scripts\pip.exe" install -e .
if errorlevel 1 ( echo Install failed. See the messages above. & pause & exit /b 1 )

:run
echo Starting Open-Sym-EPR at http://localhost:8501  (close this window to stop)
call ".venv\Scripts\streamlit.exe" run streamlit_app.py
