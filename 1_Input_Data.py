import streamlit as st
from streamlit import session_state as ss
from api.functions import *
import pandas as pd
import io
#import io

st.set_page_config(
    page_title="Data Input",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon='📊'
)

add_logo()

#   #   #   #   #   #   #   #
# Session State Management - 
if 'file_up' not in ss: # for weekly file
    ss.file_up = False
#   #   #   #   #   #   #   #

st.markdown("<h1 style='text-align: center;'>Data Input</h1>", unsafe_allow_html=True)

st.markdown("---")

st.markdown("<h3 style='text-align: left;'><u>Please read the following instructions carefully</u></h3>", unsafe_allow_html=True)
st.markdown(
"""
• Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n
"""
)

st.markdown("---")

weekly_file = st.file_uploader('Upload Weekly Sales',type='xlsx')

with st.expander('File Template Here 👇'):

    st.write('Some additonal context text here')

    excel_format_df = pd.DataFrame(columns = ['date','volume'])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine = 'xlsxwriter') as writer:
        excel_format_df.to_excel(writer,sheet_name='Sheet1',index=False)
    st.download_button(
        data = buffer,
        label = '⏬ Download Template file',
        help ='make sure your dates are continuous for best model fitting',
        use_container_width= True
    )

if (weekly_file is not None):
    ss.file_up = True
    ss['weekly_file'] = weekly_file


# Once Input Data has been uploaded :
if ss.file_up:

    #reading weekly data- [Note : Processing Being Done is here is just for Display!]
    weekly = pd.read_excel(
        ss['weekly_file'],sheet_name = 'Sheet1',
        #dtype = str
    )
    weekly['date'] = pd.to_datetime(weekly['date']).dt.date
    
    ###
    c1,c2 = st.columns(2)
    # Displaying Table
    with c1:    
        st.subheader('Weekly Data Loaded :')
        weekly_styled = (weekly.style.format({'volume' : '{:,.2f}'}))
        st.dataframe(weekly_styled,use_container_width=True,hide_index=True,height=200)
        
    # Stats -
    with c2:
        st.subheader('Input Data Stats - ')
        statistic = ['# Of Data Points','Starting','Ending']
        value = [
            len(weekly['date'].unique()),
            weekly['date'].min(),
            weekly['date'].max(),
        ]
        st.dataframe(
            data = {'Statistic' : statistic,'Value' : value}
            ,use_container_width=True
        )
    
    # Validations
    st.markdown('---')
    st.markdown(f"<h4 style='text-align: left;'><u>Validations : </u></h4>", unsafe_allow_html=True)

    if (weekly.isnull().values.any()):
        st.error("Null Values Detected !")
    else:
        st.write("No Null Values found :white_check_mark:")

    ###
    # Next Page & Store Data:
    if st.button("Next Page",use_container_width=True):
        ss['weekly'] = weekly
        st.switch_page('pages/2_Forecasting.py')