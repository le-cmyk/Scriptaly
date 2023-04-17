import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns




def choose_target():

    columns_name=list(st.session_state.cache_data.columns)
    target_column = st.selectbox(
                "Choose the target column:",
                [None]+columns_name
            )
    if target_column is not None:
        
        st.session_state.target = st.session_state.cache_data[target_column]
        st.session_state.data_modificated = change_data(st.session_state.cache_data.drop(target_column, axis=1))
    
    return target_column

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


def launch_PCA(variance_explained,data):
    # Calcul de l'ACP

    data.fillna(data.mean(), inplace=True)
    pca = PCA()
    pca.fit(data)

    nb_components = np.argmax(variance_explained > 0.95) + 1

    df_loadings = pd.DataFrame(pca.components_[:nb_components, :], columns=data.columns, index=[f'PC{i}' for i in range(1, nb_components+1)])

    # Liste des composantes à garder
    components_to_keep = list(range(1, nb_components+1))



    # Trouver le nombre de composantes principales à garder
    variance_cumsum = np.cumsum(pca.explained_variance_ratio_)
    fig = plt.figure(figsize=(12, 8))
    plt.axvline(x=np.argmax(variance_cumsum > 0.95)+1, color='red', linestyle='--')
    plt.text(np.argmax(variance_cumsum > 0.95)+1, 0.9, '95% seuil', rotation=90, va='top', ha='center', color='red')
    st.pyplot(fig)


    st.write("Composantes principales à garder pour une variance expliquée de 95% :", components_to_keep)

    # Affichage des chargements des variables sur les composantes principales sous forme de matrice de chaleur avec Seaborn
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(df_loadings.T, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
    plt.xlabel('Composante Principale')
    plt.ylabel('Variable')
    plt.title(f'Chargements des variables sur les {nb_components} premières composantes principales')
    st.pyplot(fig)


def part_preparation_data_set():

    target_column=choose_target()

    if target_column is not None:


        variance_explained =st.slider('Variance explained with PCA', min_value =0.0, max_value =1.0, value =0.95)

        launch_PCA(variance_explained,st.session_state.data_modificated)


