# to run the app : streamlit run app.py
# to have the correct version  : pipreqs --encoding=utf8 --force

import streamlit as st
import pandas as pd

#---- Importation of the class and functions
from file_code.load import Load_dataframe
from file_code.download import download_final_file,creation_code
from file_code.type_replace import Type_Replace
from file_code.visualisations import Visualisation_dataframe,Visualisation_dataframe_type,refresh_page


# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/ 
st.set_page_config(page_title="Scriptaly", page_icon="⚒", layout="wide")
path="data.csv"


if "print" not in st.session_state:
    st.session_state.print = {
        "load":"",
        "Modifications":{}
    }

if "cache_data" not in st.session_state:
    st.session_state.cache_data = []

# Top of the page
col_1,col_2=st.columns([3,0.8])
with col_1:
    st.title("Scriptaly")

with col_2:
    refresh_page()
st.markdown("#") 

# Empty space for the info panel
if "info_panel" not in st.session_state:
        st.session_state.info_panel = st.empty()


# Load a dataframe in an expander in the sidebar
Load_dataframe(path)

# Visualisation of the dataframe in an expander box
Visualisation_dataframe()       
    
# Create a button to show the DataFrame information in the sidebar
Visualisation_dataframe_type()

#Change type and replace part
Type_Replace()

st.markdown("""---""")

col_1,col_2,col_3=st.columns([0.9,1.3,1])

with col_1:

    # Add the button to your Streamlit app
    download_final_file()

with col_2:

    # Ajouter un bouton pour effacer toutes les données préenregistrées
    if st.button("Delate all the data stored in the cache"):
        st.session_state.print = {
            "load":"",
            "Modifications":{}
        }
        st.session_state.cache_data=pd.DataFrame()
        st.success("All the data were earase succesfully")

with col_3:
    # Download data as CSV
    st.download_button(
        label="Download data as CSV",
        data=st.session_state.cache_data.to_csv().encode('utf-8'),
        file_name='data_convert.csv',
        mime='text/csv',
    )

#Display the code snippet

st.code(creation_code(), language='python')

#Links

LIEN = {
    "Léo Dujourd'hui": "https://leo-dujourd-hui-digital-cv.streamlit.app",
}
SOURCES ={
    "Github": "https://github.com/le-cmyk/Scriptaly"
}
APP={
    "WebApp": "https://scriptaly.streamlit.app/"
}

col_1,col_2,col_3=st.columns([2,2,0.8])
with col_1:
    for clé, link in SOURCES.items():
        st.write(f"[{clé}]({link})")
with col_2:
    for clé, link in LIEN.items():
        st.write(f"Made by : [{clé}]({link})")
with col_3:
    for clé, link in APP.items():
        st.write(f"[{clé}]({link})")

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
