#Script to create a flask api to predict the probality of a customer
#using the model trained located in the folder model_and_data

# Path: src/app.py

import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask
from flask_restful import Api, Resource, reqparse
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = str(PROJECT_ROOT.parent)

APP = Flask(__name__)
API = Api(APP)

model = pickle.load(open(PROJECT_ROOT + '/model_and_data/model.pkl', 'rb'))

class Predict(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('SK_ID_CURR', type=int)
        args = parser.parse_args()
        test_df = pd.read_csv(PROJECT_ROOT + '/model_and_data/test_df.csv', index_col=[0])
        
        SK_ID_CURR = args['SK_ID_CURR']

        row = test_df[test_df.index == int(SK_ID_CURR)]
        
        prediction = model.predict_proba(row)
        output = np.round((1 - prediction[:,1][0])*100,1)
        return {'probability': output}

API.add_resource(Predict, '/predict')

if __name__ == "__main__":
    APP.run(debug=True, port='1080')
