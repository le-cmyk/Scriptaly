import streamlit as st
import pandas as pd


def findandreplace(data,filter_column,number_visu):

    old_value = st.text_input("Entrez la valeur à remplacer :")
    new_value = st.text_input("Entrez la nouvelle valeur :")

    # Replace the value in the specific colomne
    visualisation=data[filter_column].apply(lambda x: str(x).replace(str(old_value), str(new_value)))
    st.dataframe(pd.DataFrame(visualisation.unique()[:number_visu],columns=[filter_column]))
    
    if st.button("Save the code",key =2):

        # To find the correct name of the column 
        i=1
        while f"{filter_column}_{i}" in st.session_state.print["Modifications"].keys():
            i+=1
        replace_name=f"{filter_column}_{i}"
        replace_code=f"data['{filter_column}'] = data['{filter_column}'].apply(lambda x: str(x).replace(str('{old_value}'), str('{new_value}')))"

        if {"code":replace_code,"type":"Replacement"} not in st.session_state.print["Modifications"].values():
            st.session_state.print["Modifications"][replace_name]={"code":replace_code,"type":"Replacement"}
            st.session_state.cache_data[filter_column] = visualisation
            st.success("Replacement done and save in the code snipet")
        else:
            st.write("The filter already exist")
