import streamlit as st


# Create a button to download the filter.py file
def download_final_file():

    st.download_button(label="Download code", data=creation_code(), file_name="filter.py", mime="text/plain")





def creation_code():
    code=f"""
##Automatic creation of code
##Disclaimer the version of pandas may not be actual
    
#Importations
import pandas as pd
"""


    code+="\n"+"#Load section"
    code+="\n"+st.session_state.print['load']
    code+="\n\n"+"#Modifications"+"\n"

    for key, value in st.session_state.print['Modifications'].items():
        colonne,number_modification=key.rsplit("_", 1)
        code+="\n"+f"## {colonne} modification {number_modification} : {value['type']}"
        code+="\n"+f"{value['code']}"
    return code
