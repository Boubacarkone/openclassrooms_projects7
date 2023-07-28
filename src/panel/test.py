import pandas as pd
import numpy as np
import requests
from io import StringIO

def read_csv_from_azure(relatif_path:str):

    # URL SAP de l'objet blob
    if 'data' in relatif_path:
        sap_blob_url = "https://dashboardd.blob.core.windows.net/dashboarddata/" + relatif_path
    else:
        sap_blob_url = "https://dashboardd.blob.core.windows.net/dashboarddata/model_and_data/" + relatif_path

    # Récupération du fichier csv
    r = requests.get(sap_blob_url)

    # Création du dataframe
    df = pd.read_csv(StringIO(r.text), index_col=[0])
    return df

df = read_csv_from_azure('golbal_feature_importance.csv')
print(df.shape)

df = read_csv_from_azure('test_df_not_norm.csv')
print(df.shape)

df = read_csv_from_azure('test_df.csv')
print(df.shape)

read_csv_from_azure('data/tables/HomeCredit_columns_description.csv').reset_index()
print(df.shape)