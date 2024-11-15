import streamlit as st
from streamlit import session_state as ss
from api.functions import *

st.set_page_config(
    page_title="Short Term Forecasting Tool",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon='🔢'
)

add_logo()

st.markdown(
    "<h1 style='text-align: center;'>Short Term Forecasting</h1>", 
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown("<h3 style='text-align: left;'><u>Please read the following instructions carefully</u></h3>", unsafe_allow_html=True)
st.markdown(
"""
• Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n
"""
)