import streamlit as st
import io

def Visualisation_dataframe():

    """
    Display the DataFrame in an expander and also reset the visualisation with a button

    """
    with st.expander("Visualisation data"):

        #if st.session_state.reset_visualisation :
        st.write(st.session_state.cache_data)
        st.session_state.reset_visualisation=False

        

def Visualisation_dataframe_type():
    """Display the information of the type of the dataframe in the sidebar if the user push a button"""

    if st.sidebar.button("Type of the columns"):
        buffer_info = io.StringIO()
        st.session_state.cache_data.info(buf=buffer_info)
        st.sidebar.write("DataFrame information:")
        st.sidebar.text(buffer_info.getvalue())

def refresh_page():
    st.markdown("#")
    if st.button("Refresh"):
        var=1
