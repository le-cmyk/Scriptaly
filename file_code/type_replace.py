import streamlit as st
import pandas as pd 

from file_code.type_replace_part_type import change_type
from file_code.type_replace_part_replace import findandreplace

def Type_Replace():
    """
    The goal is to divide the page in two and propose to change the type and also replace some words by others
    """
    with st.expander("Click to filter a column"):
        columns_name=list(st.session_state.cache_data.columns)
        filter_column = st.selectbox(
                "Select column to filter:",
                [None]+columns_name
            )
        if filter_column != None:
            col_1,col_2,col_3=st.columns([2,2,1.2])

            with col_1:
                n=st.session_state.cache_data[filter_column].nunique()
                number_visu=st.slider('Number visualisation', 0, n, n)
                st.dataframe(pd.DataFrame(st.session_state.cache_data[filter_column].unique()[:number_visu],columns=[filter_column]))

            with col_3:
                # Changement of type section
                change_type(st.session_state.cache_data,filter_column)

            with col_2:

                # Replacement section
                findandreplace(st.session_state.cache_data,filter_column,number_visu)
            

