import streamlit as st
from config import PAGE_CONFIG, inject_css
from components.main_ui import run_app

def main():
    st.set_page_config(**PAGE_CONFIG)
    inject_css() # Injects modular CSS from config
    run_app()

if __name__ == "__main__":
    main()
