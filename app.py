import importlib
import streamlit as st
from config import PAGE_CONFIG, inject_css

# Force-reload submodules on every Streamlit script run so that stale
# in-memory bytecode (from sys.modules) never masks source-code fixes.
# Reload order matters: leaf modules before their importers.
import components.system_dashboard as _sys_dash
import components.main_ui as _main_ui
importlib.reload(_sys_dash)
importlib.reload(_main_ui)

from components.main_ui import run_app


def main():
    st.set_page_config(**PAGE_CONFIG)
    inject_css()
    run_app()


if __name__ == "__main__":
    main()
