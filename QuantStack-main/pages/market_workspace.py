"""The merged chart and research workspace within QuantStack."""
import os
from pathlib import Path
import sys

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title='Market workspace | QuantStack', page_icon='📈', layout='wide')
st.title('Market workspace')
st.caption('Interactive charts, stock screening, strategy testing, and Markov chain analysis.')

workspace_url = os.environ.get('QUANTSTACK_WORKSPACE_URL')
snapshot_url = os.environ.get('QUANTSTACK_SNAPSHOT_URL', 'https://hsiantw.github.io/market-atlas/')
url = workspace_url or snapshot_url
if os.environ.get('QUANTSTACK_WORKSPACE_MODE') != 'stored':
    st.info('Published daily data: charts, drawings, comparisons, screening, strategy tests, and Markov analysis. '
            'Hourly bars and the full indicator library require a connected stored dataset.')
else:
    st.caption('Reading the shared QuantStack dataset. Use Refresh inside the workspace after a collection completes.')

if workspace_url:
    components.iframe(url, height=920, scrolling=True)
    st.link_button('Open workspace in a full tab', url)
else:
    # Render services created before the merge may still launch Streamlit directly.
    # Use the merged UI here too, rather than embedding an older published UI.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.workspace_embed import snapshot_html
    components.html(snapshot_html(snapshot_url), height=920, scrolling=True)

with st.expander('Workspace guide'):
    st.markdown('''
    - **Charts:** search an asset, choose a date range, add indicators and drawings, or compare symbols.
    - **Stock screener:** open the bottom tab to filter and rank stocks, then select a result to chart it.
    - **Strategy tester:** choose a preset or custom rules, set costs, and run a backtest.
    - **Markov analysis:** fit market states, inspect transitions, forecast, and validate the model.
    - **Exports:** download price bars, screener matches, trades, chart images, or Markov results.

    Favorites, drawings, and workspace settings are saved in this browser.
    ''')
