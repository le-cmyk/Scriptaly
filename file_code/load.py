import streamlit as st
import pandas as pd
import chardet
import chardet
import io

def Load_dataframe(path):
    """
    Load a Dataframe using or a preload dataframe or the user have the possibility to drag and drop a file 
    
    """
    if "affiche_load" not in st.session_state:
        st.session_state.affiche_load = True
    if st.sidebar.button("Load section") or st.session_state.affiche_load :
        with st.sidebar:
            loadsection(path)


def loadsection(path):
    st.session_state.affiche_load = True
    data_source = st.selectbox(
        "Select data source:",
        ("Test","CSV", "Excel", "SQL")
    )

    

    if data_source=="Test":
        st.session_state.cache_data=pd.read_csv(path)

        st.session_state.print["load"]="data = pd.read_csv('"+path+"')"

    
    elif data_source in ["CSV", "Excel"]:
        uploaded_file = st.file_uploader("Upload Orders",type=['csv',"xlsx","xls"],accept_multiple_files=False)
        st.write(uploaded_file.type)
        if uploaded_file is not None:
            file_contents = uploaded_file.read()

            st.write(f"Nom du fichier : {uploaded_file.name}")

            #detect the encoding of the file
            codec = chardet.detect(file_contents)['encoding']

            try:
                # Decode the bytes-like object to a string buffer
                decoded_content = file_contents.decode(codec)
            except UnicodeDecodeError:
                # If decoding fails, try using an alternative codec
                decoded_content = file_contents.decode('utf-8', errors='replace')

            # Create a StringIO object from the string buffer
            string_buff = io.StringIO(decoded_content)

            # Read CSV from the StringIO object and display it
            if data_source == "CSV":
              st.session_state.cache_data=load_csv(string_buff,uploaded_file.name)

            elif data_source == "Excel":
              
                excel_file = pd.ExcelFile(file_contents)
                st.session_state.cache_data = excel_file.parse(excel_file.sheet_names[0])
                #data = pd.read_excel(string_buff)
                """file_contents = uploaded_file.read()
codec = chardet.detect(file_contents)['encoding']
try:
    # Decode the bytes-like object to a string buffer
    decoded_content = file_contents.decode(codec)
except UnicodeDecodeError:
    # If decoding fails, try using an alternative codec
    decoded_content = file_contents.decode('utf-8', errors='replace')
string_buff = io.StringIO(decoded_content)
excel_file = pd.ExcelFile(file_contents)
data = excel_file.parse(excel_file.sheet_names[0])

"""

                st.session_state.print["load"]="data = pd.read_excel('"+uploaded_file.name+"')"

    else :
        connection_code = st.text_area("Enter SQL connection code:")
        # Execute user's SQL connection code and check if df exists
        try:
            exec(connection_code, globals())
            if 'df' in globals():

                st.session_state.cache_data=df

                st.session_state.print["load"]=connection_code +"\ndata = df "
            else:
                st.error("The variable 'df' was not found in your code.")
        except Exception as e:
            st.error(f"Error executing SQL connection code: {e}")

    if st.button("Save Importation"):
        st.session_state.affiche_load = False
    else:
        st.session_state.info_panel.info("You forget to push the Save Importation button to save your data")



def load_csv(string_buff,name):
    data = pd.read_csv(string_buff)
    st.session_state.print["load"]="data = pd.read_csv('"+name+"')"
    return data