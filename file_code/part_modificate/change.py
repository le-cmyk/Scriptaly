import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go

from file_code.general_functions import get_function_source_code,add_new_code_section,add_new_session_state_element



def choose_target(data,target_column):

    if target_column is not None:
        
        target = data[target_column]
        data_modificated = data.drop(target_column, axis=1)

        return target,data_modificated
    else:
        return pd.DataFrame(),pd.DataFrame()



def transform_data(data,number_max_feature=100):
    """
    Preprocesses data by normalizing numeric features and one-hot encoding categorical features with less than number_max_feature unique values.
    Returns a DataFrame of the transformed data.
    """
    # Identify numeric and categorical features
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    # Create transformers to normalize numeric features and perform one-hot encoding on categorical features
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Select only columns with less than number_max_feature unique values for one-hot encoding
    categorical_features_selected = []
    for feature in categorical_features:
        if data[feature].nunique() < number_max_feature:
            categorical_features_selected.append(feature)
    
    # Convert selected categorical columns to string type
    data[categorical_features_selected] = data[categorical_features_selected].astype(str)

    # Create a preprocessor that applies the appropriate transformers to the appropriate columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features_selected)])

    # Fit the categorical transformer to create the mapping of categories
    categorical_transformer.fit(data[categorical_features_selected])

    # Apply the preprocessor to the data
    data_transformed = preprocessor.fit_transform(data)

    # Get column names for the categorical variables
    categorical_names = categorical_transformer.get_feature_names_out(categorical_features_selected)
    feature_names = list(numeric_features) + list(categorical_names)

    # Convert the result to a DataFrame
    data_transformed_df = pd.DataFrame(data_transformed.toarray(), columns=feature_names)

    return data_transformed_df



def visualisation_pca(variance_explained,columns,pca):

    variance_cumsum = np.cumsum(pca.explained_variance_ratio_)
    nb_components = np.argmax(variance_cumsum > variance_explained) + 1
    df=pca.components_[:nb_components, :]
    index=[f'PC{i}' for i in range(1, nb_components+1)]

    df_loadings = pd.DataFrame(df, columns=columns, index=index)

    # Calcul des valeurs pour le graphique
    variance_ratio = pca.explained_variance_ratio_

    col1,col2=st.columns([0.7,1.1])

    # Création de la figure avec Plotly GO
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(variance_ratio)+1)), y=variance_ratio, mode='markers+lines', marker=dict(symbol='circle', size=8, color='blue'), name='Variance expliquée'))
    fig.add_trace(go.Scatter(x=list(range(1, len(variance_cumsum)+1)), y=variance_cumsum, mode='markers+lines', marker=dict(symbol='circle', size=8, color='green'), name='Variance cumulée'))
    fig.add_vline(x=nb_components, line_dash='dash', line_color='red',
                  annotation_text=f' Seuil {variance_explained}% col n°{nb_components}', annotation_position='right')
    fig.update_layout(xaxis_title='Composante Principale', 
                      yaxis_title='Variance expliquée (%)', 
                      title='PCA informations',
                      legend=dict(
                            x=0.5,
                            y=0.8,
                            traceorder='normal',
                        ),
                     width=400)
    col1.write(fig)


    fig = go.Figure(data=go.Heatmap(
                   z=df_loadings.T.values,
                   x=df_loadings.index,
                   y=columns,
                   colorscale='RdBu_r',
                   zmin=-1,
                   zmax=1,
                   showscale=True,
                   showlegend=False,
                   colorbar=dict(title='Loading')
                 ))
    fig.update_layout(
        title=f'Loadings of variables on the first {nb_components} principal components' ,
        height=20*len(columns))

    annotations = []
    for y in range(len(columns)):
        for x in range(len(df_loadings.index)):
            annotations.append(
                dict(
                    x=df_loadings.index[x],
                    y=columns[y],
                    text=f"{df_loadings.T.values[y][x]:.2f}",
                    showarrow=False,
                    font=dict(size=10, color='black')
                )
            )

    fig.update_layout(annotations=annotations)

    # Display heatmap in Streamlit app
    col2.plotly_chart(fig)
    
    col_1,col_2,col_3=st.columns([1,1.5,0.6])
    full_col=st.columns(1)
    with col_1:
        if st.button("Diplay PCA columns"):
            full_col[0].write(df_loadings)

    with col_3:
        # Download data as CSV
        st.download_button(
            label="Download PCA columns",
            data=df_loadings.to_csv().encode('utf-8'),
            file_name='PCA_columns.csv',
            mime='text/csv',
        )




def calculate_pca(data, variance_explained_threshold):
    """
    Calculates the Principal Component Analysis (PCA) of the input data, selects the number of components
    that cumulatively explain a minimum percentage of the variance, and returns the transformed data with the 
    selected components.
    
    Args:
        data (pd.DataFrame): Input data to calculate the PCA on.
        variance_explained_threshold (float): Minimum percentage of the variance that the selected components 
                                             must cumulatively explain.
        
    Returns:
        pca (PCA object): PCA object fitted to the input data.
        data_pca (pd.DataFrame): Transformed data with the selected principal components.
    """
    # Replace missing values with the column means
    data.fillna(data.mean(), inplace=True)
    
    # Calculate the PCA
    pca = PCA()
    pca.fit(data)
    
    # Calculate the percentage of variance explained by each principal component
    variance_ratios = pca.explained_variance_ratio_
    
    # Select the indices of the columns that cumulatively explain enough variance
    variance_cumulative = np.cumsum(variance_ratios)
    n_components = np.argmax(variance_cumulative >= variance_explained_threshold) + 1
    selected_columns = range(n_components)
    
    # Transform the original data into the selected principal components
    data_pca = pca.transform(data)[:, selected_columns]
    
    # Create a new dataframe with the selected principal components
    col_names = [f'PC{i+1}' for i in selected_columns]
    df_pca = pd.DataFrame(data_pca, columns=col_names)
    
    return pca, df_pca


def check_null_values(df):
    """
    Checks if a pandas DataFrame contains null or NaN values.
    
    Args:
    ----
    df: pd.DataFrame
        The DataFrame to be checked for null or NaN values.
        
    Returns:
    -------
    bool
        True if the DataFrame does not contain null or NaN values, False otherwise.
    """
    null_cols = df.columns[df.isnull().any()].tolist()
    if len(null_cols) == 0:
        return True
    else:
        # Display an error message with the names of the columns that contain null or NaN values
        st.error(f"The following columns contain null or NaN values: {null_cols}")
        return False





def part_preparation_data_set():


    data=st.session_state.cache_data

    target_column = st.selectbox(
                "Choose the target column:",
                [None]+list(data.columns)
            )

    if target_column is not None:

        st.session_state.target,data_modificated=choose_target(data,target_column)
        col_1,col_2,col_3=st.columns([1,1.5,0.6])

        if col_1.button("Load targets"):
            add_new_code_section("Target" ,{"function":   get_function_source_code(choose_target)} )
            if ["function"] == list(st.session_state.print["Target"].keys()):  

                st.session_state.print["Target"][target_column]=f"y,x = choose_target(data,'{target_column}')"

                st.success(f"Target selectionn in the code")

        if check_null_values(data_modificated): 

            if col_2.button("Normalise and encode"):

                st.session_state.data_modificated = transform_data(data_modificated)
                add_new_code_section("Normalise_hot_encoding" ,{"function":   get_function_source_code(transform_data)} )
                
                st.session_state.print["Normalise_hot_encoding"]["Application"]=f"x = change_data(x)"

                st.success(f"Normalise and encode done and write in the code")

                    
            add_new_session_state_element("display_PCA",False)
            st.session_state.display_PCA=col_3.button("Application of PCA") != st.session_state.display_PCA

            if st.session_state.display_PCA:
                variance_explained =st.slider('Variance explained with PCA', min_value =0.0, max_value =1.0, value =0.95,step=0.002,format="%.3f")

                pca,df_pca=calculate_pca(st.session_state.data_modificated, variance_explained)

                variance_ratios = pca.explained_variance_ratio_
                variance_cumulative = np.cumsum(variance_ratios)
                n_components = np.argmax(variance_cumulative >= variance_explained) + 1
                
                st.session_state.df_pca=df_pca.iloc[:,:n_components]

                add_new_code_section("PCA" ,{"function":   get_function_source_code(calculate_pca)} )
                st.session_state.print["PCA"]["Application"]=f"pca,x=calculate_pca(x, {variance_explained})"+"\n"+"variance_ratios = pca.explained_variance_ratio_"+"\n"+"variance_cumulative = np.cumsum(variance_ratios)"+"\n"+"n_components = np.argmax(variance_cumulative >= variance_explained) + 1"+"\n"+"x=x.iloc[:,:n_components]"

                st.success(f"PCA done")

                visualisation_pca(variance_explained,st.session_state.data_modificated.columns,pca)




