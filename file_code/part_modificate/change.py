import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go




def choose_target(data):

    columns_name=list(data.columns)
    target_column = st.selectbox(
                "Choose the target column:",
                [None]+columns_name
            )
    if target_column is not None:
        
        target = data[target_column]
        data_modificated = data.drop(target_column, axis=1)

        return target_column,target,data_modificated
    else:
        return target_column,pd.DataFrame(),pd.DataFrame()

def normalisation_data(X):
    # Normalisation of data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def change_data(data):
    # Identifier les variables numériques et catégorielles
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    
    # Créer les transformateurs pour normaliser les données numériques et effectuer l'encodage one-hot des données catégorielles
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Sélectionner uniquement les colonnes avec moins de 100 valeurs uniques pour l'encodage one-hot
    categorical_features_selected = []

    for feature in categorical_features:
        if data[feature].nunique() < 100:
            categorical_features_selected.append(feature)
    data[categorical_features_selected]=data[categorical_features_selected].astype(str)

    # Créer un preprocessor qui applique les transformateurs aux colonnes appropriées
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features_selected)])

    # Adapter le transformateur aux données catégorielles pour créer le mapping des catégories
    categorical_transformer.fit(data[categorical_features_selected])

    # Appliquer le preprocessor aux données
    data_transformed = preprocessor.fit_transform(data)

    # Obtenir les noms des colonnes pour les variables catégorielles
    categorical_names = categorical_transformer.get_feature_names_out(categorical_features_selected)
    feature_names = list(numeric_features) + list(categorical_names)

    # Convertir le résultat en DataFrame
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




def calcul_pca(data, variance_explained):
    # Calcul de l'ACP
    data.fillna(data.mean(), inplace=True)
    pca = PCA()
    pca.fit(data)
    
    # Calculer le pourcentage de variance expliquée par chaque composante principale
    variance_ratios = pca.explained_variance_ratio_
    
    # Sélectionner les indices des colonnes qui cumulent suffisamment de variance
    variance_cumulative = np.cumsum(variance_ratios)
    n_components = np.argmax(variance_cumulative >= variance_explained) + 1
    selected_columns = range(n_components)
    
    # Transformer les données d'origine en composantes principales sélectionnées
    data_pca = pca.transform(data)[:, selected_columns]
    
    # Créer un nouveau dataframe avec les composantes principales sélectionnées
    col_names = [f'PC{i+1}' for i in selected_columns]
    df_pca = pd.DataFrame(data_pca, columns=col_names)
    
    return pca,df_pca

def check_null_values(df):
    """
    Vérifie si un dataframe contient des valeurs null ou nan.
    Retourne True si le dataframe ne contient pas de valeurs null ou nan, False sinon.
    """
    null_cols = df.columns[df.isnull().any()].tolist()
    if len(null_cols) == 0:
        return True
    else:
        st.error(f"Les colonnes suivantes contiennent des valeurs nulles ou NaN : {null_cols}")
        return False




def part_preparation_data_set():

    target_column,st.session_state.target,data_modificated=choose_target(st.session_state.cache_data)

    if target_column is not None and check_null_values(data_modificated):

        data_modificated = change_data(data_modificated)
        variance_explained =st.slider('Variance explained with PCA', min_value =0.0, max_value =1.0, value =0.95,step=0.002,format="%.3f")

        pca,df_pca=calcul_pca(data_modificated, variance_explained)

        variance_ratios = pca.explained_variance_ratio_
        variance_cumulative = np.cumsum(variance_ratios)
        n_components = np.argmax(variance_cumulative >= variance_explained) + 1
        st.write(n_components)
        st.session_state.df_pca=df_pca.iloc[:,:n_components]
        visualisation_pca(variance_explained,data_modificated.columns,pca)



