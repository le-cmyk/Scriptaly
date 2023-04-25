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

    for key in st.session_state.print.keys():
        if key=='Modifications':
            code+="\n\n"+"#Modifications"+"\n"
            for key2, value in st.session_state.print[key].items():
                colonne,number_modification=key2.rsplit("_", 1)
                code+="\n"+f"## {colonne} modification {number_modification} : {value['type']}"
                code+="\n"+f"{value['code']}"
        else:
            code+="\n\n"+f"#{key} section"
            if type(st.session_state.print[key])==str:
                code+="\n"+st.session_state.print[key]
            elif type(st.session_state.print[key])==dict:
                show_function=True
                for key2, value in st.session_state.print[key].items():
                    if key2!="function":
                        code+="\n"+f"## {key2}"
                        code+="\n"+f"{value}"

                    elif show_function:
                        show_function=False
                        code+="\n"+f"## {key2}"
                        code+="\n"+f"{value}"

    
    return code
