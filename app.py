import os

# Limit TF to 2 threads so extraction/training on the Pi 4 doesn't starve
# the Streamlit server thread of CPU time. Must be set before TF is imported.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import streamlit as st

from config import PAGE_CONFIG, inject_css

from components.main_ui import run_app


def main():
    st.set_page_config(**PAGE_CONFIG)
    inject_css()
    run_app()


if __name__ == "__main__":
    main()
