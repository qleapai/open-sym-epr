"""Streamlit Cloud entry point for Open-Sym-EPR.

Runs the main simulation-and-fitting application. Streamlit Cloud (and
`streamlit run streamlit_app.py`) executes this file; it simply runs the app.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "app_simepr.py"), run_name="__main__")
