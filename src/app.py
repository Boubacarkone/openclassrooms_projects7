#Script to create a flask api to predict the probality of a customer
#using the model trained located in the folder model_and_data

# Path: src/app.py

import os
import pickle
import pandas as pd
import numpy as np
import json
import re
import flask
from flask import Flask
#from flask_cors import CORS
from flask_restful import Api, Resource, reqparse
from pathlib import Path

import lime
import lime.lime_tabular
import time

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = str(PROJECT_ROOT.parent)

APP = Flask(__name__)
#CORS(APP)
API = Api(APP)


model = pickle.load(open(PROJECT_ROOT + '/model_and_data/model.pkl', 'rb'))

test_df = pd.read_csv(PROJECT_ROOT + '/model_and_data/test_df.csv', index_col=[0])


X_train = pd.read_csv(PROJECT_ROOT + '/model_and_data/X_train.csv', index_col=[0])
X_test = pd.read_csv(PROJECT_ROOT + '/model_and_data/X_test.csv', index_col=[0])
y_train = pd.read_csv(PROJECT_ROOT + '/model_and_data/y_train.csv', index_col=[0])
y_test = pd.read_csv(PROJECT_ROOT + '/model_and_data/y_test.csv', index_col=[0])
print("Data loaded:")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Get the index of the categorical variables
var_cats = []
for var in test_df.columns:

    if len(test_df[var].value_counts().values.tolist()) == 2:
        var_cats.append(var)

print(len(var_cats))
var_cats_idx = [test_df.columns.to_list().index(cat) for cat in var_cats]

# LIME has one explainer for all the models
explainer = lime.lime_tabular.LimeTabularExplainer(
    
    X_train.values, feature_names=X_train.columns.values.tolist(),
    class_names=['TARGET'], verbose=False, mode='classification',
    random_state=42, categorical_features=var_cats_idx
)

def local_explainer(customer_id, explainer):

    exp = explainer.explain_instance(
        test_df[test_df.index == customer_id].values[0],
        predict_fn = model.predict_proba,
        num_features=len(X_train.columns)
    )

    exp_df = pd.DataFrame(exp.as_list(), columns=['Feature', 'Importance'])
    exp_df['abs_values'] = np.abs(exp_df.Importance)
    exp_df.sort_values(by='abs_values', ascending=True, inplace=True)
    #exp_df.set_index('Feature', inplace=True)
    #exp_df.drop('abs_values', axis=1, inplace=True)
    return exp_df


# Controller-1
@APP.route("/demo", methods=['GET'])
def get_demo():
    return "This is a demo api"

# Controller-2
@APP.route("/predict", methods=['POST'])
def get_demo_name():
    data = flask.request.get_data()
    print(str(data), 'True' in str(data))
    if not data:
        return "No data provided."
    else:
        regex = re.compile(r'\d+')
        name = regex.findall(str(data))[0]
        SK_ID_CURR = int(name)

        try:
            row = test_df[test_df.index == int(SK_ID_CURR)]
        except Exception as e:
                return f'Error to get test row data! : {e}'

        if 'True' in str(data):
            print("local_feature_importance_show = True")

            try:
                start_time = time.time()
                exp_df = local_explainer(SK_ID_CURR, explainer)
                time_elapsed = time.time() - start_time
            except Exception as e:
                return f'Error to get the local feature importance! : {e}'

        else:
            print("local_feature_importance_show = False")
            exp_df = pd.DataFrame()
            time_elapsed = 0

        try:
            prediction = model.predict_proba(row)
            output = np.round((1 - prediction[:,1][0])*100,1)
        except Exception as e:
            return f"Model prediction error : {e}"

        return {
        'Trust rate': output,
        'predict_proba': prediction[:,1][0],
        'local_explainer_df': exp_df.to_dict(),
        'time_elapsed': time_elapsed
        }

#Get the categorical variables data to be used in the front end
if __name__ == "__main__":
    APP.run(debug=True, port=8000)

#gunicorn -w 4 --bind 0.0.0.0:8000 src/app:app
