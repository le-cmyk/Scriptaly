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

        col_1,col_2=st.columns([2.5,0.5])
        filter_column = col_1.selectbox(
                "Select column to filter:",
                [None]+columns_name
            )
        st.session_state.display_drop_na = col_2.button("Drop all na")

        if filter_column != None:

            if "display_modifications" not in st.session_state:
                st.session_state.display_modifications=False
            col_1,col_2,col_3=st.columns([2,2,1.2])

            st.session_state.display_modifications = col_1.button("Modifications") != st.session_state.display_modifications
            st.session_state.display_delate_column = col_3.button("Delate column") 
            st.session_state.display_delate_lines = col_2.button("Delate lines")
            
            if st.session_state.display_modifications:
                st.write("---")

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
            
            elif st.session_state.display_delate_column:

                st.write("---")
                if "Delation" not in st.session_state.print.keys():
                    st.session_state.print["Delation"]={}
                if filter_column not in st.session_state.print["Delation"].keys():
                    st.session_state.cache_data = st.session_state.cache_data.drop(filter_column, axis=1)
                    st.session_state.print["Delation"][filter_column]=f"data = data.drop('{filter_column}', axis=1)"
                    st.success(f"Delation of the {filter_column} column done")
            elif st.session_state.display_delate_lines:
                
                st.write("---")
        elif st.session_state.display_drop_na:
            if "Drop all na" not in st.session_state.print.keys():
                #Only if it's not already drop
                st.session_state.print["Drop all na"]=""
                st.session_state.cache_data=st.session_state.cache_data.dropna()
                st.session_state.print["Drop all na"]="data=data.dropna()"
                st.success("All na line droped")
            else : 
                st.error("Drop all na previously done")


            



               




            

