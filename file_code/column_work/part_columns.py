import streamlit as st
import plotly.graph_objects as go

from file_code.general_functions import get_function_source_code,add_new_code_section,add_new_session_state_element

def stats_on_columns():
    with st.expander("Select columns to slice"):
        df=st.session_state.cache_data
        # Create a list of column names with NaN as an option
        columns = list(df.columns)

        # Create a checkbox for each column in the dataframe
        selected_columns = st.multiselect('Select columns to slice', columns, default=[],label_visibility="collapsed")
        
        # Filter the dataframe based on the selected columns
        if 0!= len(selected_columns):
            filtered_df = df[selected_columns]


            for column in selected_columns:
                st.write("---")
                df_filter=filtered_df[column]
                previous_length=df_filter.shape[0]

                col_1,col_2=st.columns([1,1])
                with col_1:
                    affichage_histogram(df_filter,column,f"Before selection distribution of {column}")

                slider_key="quartile"+column
                add_new_session_state_element(slider_key,(0,0))
                q_low,q_high=st.slider('Remove outliers', 0.0, 100.0, (5.0, 95.0),key=slider_key)
                df_filter=remove_outliers(df_filter,q_low,q_high)
                with col_2:
                    affichage_histogram(df_filter,column,f"After selection distribution of {column}")

                if st.button("Apply changes",key=column):
                    add_new_code_section("Slice_column" ,{"function":   get_function_source_code(remove_outliers)+"\n"+
                                                                        get_function_source_code(filter_dataframe)} )
                    if column not in st.session_state.print["Slice_column"].keys():      
                        st.session_state.cache_data=filter_dataframe(st.session_state.cache_data,df_filter)

                        st.session_state.print["Slice_column"][column]=f"column_filter = remove_outliers(data['{column}'],{q_low},{q_high})"+"\n"+f"data = filter_dataframe(data,column_filter)"

                        st.success(f"In total {previous_length-st.session_state.cache_data.shape[0]} lines where delated")
                    

def affichage_histogram(df,name,title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df, name=name))
    fig.update_layout(title=title,width=400)
    st.plotly_chart(fig)

def remove_outliers(df, lower_quantile_threshold, upper_quantile_threshold):
    """
    This function removes the outliers in a pandas dataframe column by filtering out the values
    that fall outside a specified range of quantiles.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the column to filter
    lower_quantile_threshold : float
        The lower quantile threshold as a percentage, below which values will be filtered out
    upper_quantile_threshold : float
        The upper quantile threshold as a percentage, above which values will be filtered out

    Returns:
    --------
    pandas.DataFrame
        The filtered dataframe with the outliers removed
    """

    # Calculate the lower and upper quantile thresholds
    lower_quantile_value = df.quantile(lower_quantile_threshold / 100)
    upper_quantile_value = df.quantile(upper_quantile_threshold / 100)

    # Filter the dataframe to keep only the values within the specified range of quantiles
    filtered_df = df[(df > lower_quantile_value) & (df < upper_quantile_value)]

    return filtered_df


def filter_dataframe(df1, df2):
    """
    Filter df1 based on the rows present in df2 using the indexes
    :param df1: pandas DataFrame to filter
    :param df2: pandas DataFrame to use for filtering
    :return: filtered pandas DataFrame
    """
    # Get the intersection of the indexes of df1 and df2
    index_intersection = df1.index.intersection(df2.index)

    # Filter df1 based on the intersection of the indexes
    filtered_df = df1.loc[index_intersection]

    return filtered_df

def futur():
    futur="""

import numpy as np
from scipy.stats import shapiro, anderson
from statsmodels.stats.diagnostic import lilliefors

# Test de normalité avec le test de Shapiro-Wilk
shapiro_stat, shapiro_p = shapiro(data)
if shapiro_p < 0.05:
    print("Les données ne suivent pas une distribution normale (p-value = {:.4f})".format(shapiro_p))
else:
    print("Les données suivent une distribution normale (p-value = {:.4f})".format(shapiro_p))

    
# Test de normalité avec le test d'Anderson-Darling
ad_stat, ad_critical, ad_significance = anderson(data, 'norm')
if ad_stat > ad_critical[2]:
    print("Les données ne suivent pas une distribution normale (statistique de test = {:.4f})".format(ad_stat))
else:
    print("Les données suivent une distribution normale (statistique de test = {:.4f})".format(ad_stat))

    

# Test d'indépendance et d'identique distribution
lil_stat, lil_p = lilliefors(data)
if lil_p < 0.05:
    print("Les données ne sont pas indépendantes et identiquement distribuées (p-value = {:.4f})".format(lil_p))
else:
    print("Les données sont indépendantes et identiquement distribuées (p-value = {:.4f})".format(lil_p))


    


# Calcul de la moyenne et de l'écart-type
mean = df['mesure'].mean()
std = df['mesure'].std()

# Calcul du CPK
CPK = np.min([(df['LST'] - mean) / (3 * std), (mean - df['LIT']) / (3 * std)])    
"""