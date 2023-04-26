#Test the code app.py using pytest
import pytest
import app

def app_model_load_test():
    """
    Test the model load
    """
    assert app.model is not None

def app_test_df_load_test():
    """
    Test the test_df load
    """
    assert app.test_df is not None
