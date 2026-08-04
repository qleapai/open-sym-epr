@echo off
REM Open-Sym-EPR launcher
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
    echo Creating virtual environment and installing Open-Sym-EPR...
    python -m venv .venv
    call .venv\Scripts\pip install -e .
)
echo Starting Open-Sym-EPR at http://localhost:8501 ...
call .venv\Scripts\streamlit run streamlit_app.py
