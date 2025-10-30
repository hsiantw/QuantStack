import os
import sys

import streamlit as st

# Ensure local packages can be imported when running with `streamlit run app.py`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils.auth import init_session_state, show_auth_page, show_user_menu  # noqa: E402
from pages.market_performance_tracker import main as market_performance_main  # noqa: E402

st.set_page_config(
    page_title="Market Performance Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_app() -> None:
    """Render the streamlined QuantStack experience."""
    init_session_state()

    if not st.session_state.get("authenticated", False):
        show_auth_page()
        return

    show_user_menu()
    market_performance_main()


if __name__ == "__main__":
    render_app()
