# Unit tests pour le deploiement de l'API
import pytest
import requests
import pandas as pd
from io import StringIO, BytesIO
import pickle
import imblearn.pipeline
from app import read_csv_from_azure


# test de la fonction read_csv_from_azure

def test_get_data_from_azure():
    """Test the function read_csv_from_azure"""

    df = read_csv_from_azure('X_train.csv')
    assert type(df) == pd.core.frame.DataFrame
    assert df.shape == (23062, 374)

    df = read_csv_from_azure('test_df.csv')
    assert type(df) == pd.core.frame.DataFrame
    assert df.shape == (4874, 374)

# test du chargement du model

def test_model():
    """Test the model"""

    # URL SAP de l'objet blob
    sap_blob_url = "https://dashboardd.blob.core.windows.net/dashboarddata/model_and_data/model.pkl"

    # Récupération du fichier pkl
    r = requests.get(sap_blob_url)

    # Récupération du model
    model = pickle.load(BytesIO(r.content))

    assert type(model) == imblearn.pipeline.Pipeline

#pour lancer les tests : pytest -v test.py
