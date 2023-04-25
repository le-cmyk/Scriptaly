import inspect
import streamlit as st


def get_function_source_code(function):
    """
    This function takes another function as a parameter and returns the source code of that function as a string.
    """
    source_code = inspect.getsource(function)
    return source_code

def add_new_code_section(name,element={}):
    """
    Add a new section in the code snipet if the section not exist
    """
    if name not in st.session_state.print.keys():
        st.session_state.print[name]=element

def add_new_session_state_element(name,element):
    if name not in st.session_state.keys():
        st.session_state[name]=element

