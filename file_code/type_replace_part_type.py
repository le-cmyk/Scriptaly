import streamlit as st
from dateutil.parser import parse
import pandas as pd

from file_code.general_functions import get_function_source_code



def change_type(data,filter_column):
    """
    Changes the data type of a selected column in a pandas DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame to modify.
        filter_column (str): The name of the column to modify.

    Returns:
        None.
    """

    
    filter_code=""

    # Get the selected column and display its information
    selected_column = data[filter_column]
    st.sidebar.write(f"{filter_column} information:")
    st.sidebar.write(selected_column.describe())

    # Allow the user to change the data type of the column
    list_types=["int64", "string","float64", "object", "bool","datetime64[ns]"]
    new_type = st.selectbox("Select new data type:", list_types,index=list_types.index(selected_column.dtype))

    if new_type != selected_column.dtype:
        if new_type == 'datetime64[ns]':
            if st.button("Buldozer"):
                converted_column=parse_datetime_column(selected_column,filter_column)
                filter_code+=get_function_source_code(parse_datetime_column)
                filter_code+=f"\ndata['{filter_column}'] = parse_datetime_column(data['{filter_column}'],'{filter_column}')"
            else:
                converted_column,format = convert_datetime_simpler(selected_column)
                filter_code+=f"data['{filter_column}'] = pd.to_datetime(data['{filter_column}'],format='{format}')"
        elif new_type=="bool":
            converted_column,dictionnaire=convert_to_bool(selected_column)
            filter_code+=f"dictionnaire = {str(dictionnaire)}"
            filter_code+=f"\ndata['{filter_column}'] = data['{filter_column}'].map(dictionnaire)"

        else:
            converted_column = selected_column.astype(new_type)
            filter_code+=f"data['{filter_column}'] = data['{filter_column}'].astype('{new_type}')"

        # Save the modification
        if st.button("Save the code",key =1):
            if "Modifications" not in st.session_state.print.keys():
                st.session_state.print["Modifications"]={}

            # To find the correct name of the column 
            i=1
            while f"{filter_column}_{i}" in st.session_state.print["Modifications"].keys():
                i+=1
            filter_name=f"{filter_column}_{i}"
            
            if {"code":filter_code,"type":"Type"} not in st.session_state.print["Modifications"].values():
                st.session_state.print["Modifications"][filter_name]={"code":filter_code,"type":"Type"}
                st.session_state.cache_data[filter_column] = converted_column
                st.success(f"New type {filter_column}: {st.session_state.cache_data[filter_column].dtypes} code saved and visible")
            else:
                st.write("The filter already exist")


@st.cache_data
def parse_datetime_column(column, column_name):
    """
    Parses each datetime string in a pandas dataframe column using dateutil,
    and returns a new column with parsed datetime values.

    Parameters:
        column (pandas.Series): The column to parse.
        column_name (str): The name of the column to create.

    Returns:
        pandas.DataFrame: The new column with parsed datetime values.
    """

    datetime_col = pd.DataFrame(columns=[column_name], dtype='datetime64[ns]')
    for val in column:
        try:
            datetime_val = parse(val)
            datetime_col = datetime_col.append({column_name: datetime_val}, ignore_index=True)
        except:
            datetime_col = datetime_col.append({column_name: pd.NaT}, ignore_index=True)
    return datetime_col


def convert_datetime_simpler(column):
    st.write("Write the format :")
    st.write("2018-01-02 06:00:00.516Z -> %Y-%m-%d %H:%M:%S.%fZ")
    format=st.text_input("Format","%Y-%m-%d %H:%M:%S.%fZ")
    column = col_to_datetime(column,format)
    return column,format

@st.cache_data
def col_to_datetime(column,format):
    return pd.to_datetime(column, format=format)

def convert_to_bool(column):
    """
    Ask the User for the True and False Value
    convert the column

    Parameters:
        column (pandas.Series): The column to parse.

    Returns:
        pandas.DataFrame: The new column with Bool values.
    """
    unique=column.unique()
    if "convert_bool" not in st.session_state:
        st.session_state.convert_bool = True
    if len(unique)!=2:
        st.Write("More than two different value in the column")
    else:
        if st.session_state.convert_bool :
            dictionnaire={unique[0]:True,unique[1]:False}
        else:
            dictionnaire={unique[1]:True,unique[0]:False}
        st.write(dictionnaire[unique[0]],":",unique[0])
        st.write(dictionnaire[unique[1]],":",unique[1])
        if st.button("Switch"):
            st.session_state.convert_bool = not st.session_state.convert_bool
        column=column.map(dictionnaire)
    return column,dictionnaire
        
        