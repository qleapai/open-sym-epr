@echo off
REM ============================================================
REM   EPR-Suite one-click launcher (OpenSpin + SimEPR)
REM   First run sets up a private virtual environment.
REM ============================================================
setlocal
cd /d "%~dp0"
set VENV=.venv
set PY=%VENV%\Scripts\python.exe
set STREAMLIT=%VENV%\Scripts\streamlit.exe

REM --- create the venv on first run ---
if not exist "%PY%" (
  echo [setup] Creating virtual environment ^(first run only^)...
  py -3.11 -m venv "%VENV%"
  if errorlevel 1 (
    echo [error] Python 3.11 not found. Install Python 3.11 from python.org and retry.
    pause
    exit /b 1
  )
  echo [setup] Installing dependencies ^(this may take a few minutes^)...
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r requirements.txt
)

:menu
echo.
echo ============================================================
echo                      E P R   -   S U I T E
echo ============================================================
echo   [1]  OpenSpin   - native Python EPR/ENDOR/ESEEM/magnetometry
echo   [2]  SimEPR     - cw-EPR simulation, fitting, batch, EasySpin
echo   [3]  Run tests
echo   [Q]  Quit
echo ============================================================
set /p choice="Select an option and press Enter: "

if /i "%choice%"=="1" goto openspin
if /i "%choice%"=="2" goto simepr
if /i "%choice%"=="3" goto tests
if /i "%choice%"=="Q" goto end
echo Invalid choice.
goto menu

:openspin
echo [run] Launching OpenSpin at http://localhost:8501 ...
"%STREAMLIT%" run app_openspin.py --server.port 8501
goto menu

:simepr
echo [run] Launching SimEPR at http://localhost:8502 ...
"%STREAMLIT%" run app_simepr.py --server.port 8502
goto menu

:tests
echo [test] Running the test suite...
"%PY%" -m pip install pytest -q
"%PY%" -m pytest tests -q
pause
goto menu

:end
echo Goodbye.
endlocal
