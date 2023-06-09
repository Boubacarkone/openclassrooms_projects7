import pytest
import requests
import pandas as pd
from io import StringIO

def test_api_response_1():
    """Test the api response with local_feature_importance_show = False"""

    url = 'http://flask-api.francecentral.cloudapp.azure.com:5000/predict'
    SK_ID_CURR = 265669
    data = {'SK_ID_CURR': SK_ID_CURR, 'local_feature_importance_show': False}

    response = requests.post(url, data=data)

    assert type(response) == requests.models.Response
    assert response.status_code == 200
    assert response.json() is not None

def test_api_response_2():
    """Test the api response with local_feature_importance_show = True"""

    url = 'http://flask-api.francecentral.cloudapp.azure.com:5000/predict'
    SK_ID_CURR = 265669
    data = {'SK_ID_CURR': SK_ID_CURR, 'local_feature_importance_show': True}

    response = requests.post(url, data=data)

    assert type(response) == requests.models.Response
    assert response.status_code == 200
    assert response.json() is not None


def read_csv_from_azure(relatif_path:str):
    """Read csv file from azure blob storage"""
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



def test_get_data_from_azure():
    """Test the function read_csv_from_azure"""
    
    df = read_csv_from_azure('golbal_feature_importance.csv')
    assert type(df) == pd.core.frame.DataFrame
    assert df.shape == (374, 2)

    df = read_csv_from_azure('test_df_not_norm.csv')
    assert type(df) == pd.core.frame.DataFrame
    assert df.shape == (4874, 375)

    df = read_csv_from_azure('test_df.csv')
    assert type(df) == pd.core.frame.DataFrame
    assert df.shape == (4874, 374)

    read_csv_from_azure('data/tables/HomeCredit_columns_description.csv').reset_index()
    assert type(df) == pd.core.frame.DataFrame
    assert df.shape == (4874, 374)