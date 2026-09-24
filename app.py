"""Streamlit Cloud entry point for Open-Sym-EPR (alias of streamlit_app.py).

Some deployments reference `app.py` as the main module. This file runs the same
simulation-and-fitting application, so either `app.py` or `streamlit_app.py` works
as the Streamlit Cloud main file.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "app_simepr.py"), run_name="__main__")
