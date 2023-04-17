import streamlit as st
import io

def Visualisation_dataframe():

    """
    Display all DataFrame in an expander
    """
    with st.expander("Visualisation data"):

        tab1, tab2 = st.tabs(["🗃 Data source", "🗃 Data prediction"])

        tab1.write(st.session_state.cache_data)

        col1,col2=tab2.columns([0.2,1])

        col1.write(st.session_state.target)
        col2.write(st.session_state.data_modificated)

        

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
