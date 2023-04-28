#Creat the folder data/tables if it does not exist
#create the folder model_and_data if it does not exist
#Get the data from URL and decompress it to the folder data/tables
#Preprocess the data and save it to the folder model_and_data
#Train the model and save it to the folder model_and_data
#Deploy the model using flask

import os
import pandas as pd
import src.utils.udf as udf
import src.datapreprocessing as dp
import re
from subprocess import run, PIPE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = str(PROJECT_ROOT)

print("Folder structure creation :")
#Creat the folder data/tables if it does not exist
run("mkdir -p " + PROJECT_ROOT + "/data", shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)
run("mkdir -p " + PROJECT_ROOT + "/data/tables", shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)

#create the folder model_and_data if it does not exist
run("mkdir -p " + PROJECT_ROOT + "/model_and_data", shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)

print("Data download...")
#Get the data from URL and decompress it to the folder data/tables
url = 'https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip'

#Download the file from url and save it to the folder data/tables with line command
run(f"wget -P {PROJECT_ROOT}/data {url}", shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)

print("Data preprocessing...")
#Unzip the file with line command
run(f"unzip {PROJECT_ROOT}/data/Projet+Mise+en+prod+-+home-credit-default-risk.zip -d {PROJECT_ROOT}/data/tables", shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)


#Preprocess the data and save it to the folder model_and_data

dp.data_process(PROJECT_ROOT + '/data/tables', 0, False)

#import data_preprocessed from data
data_preprocessed = pd.read_csv(PROJECT_ROOT + '/data/data_pre_processed_final_v0.csv', index_col=[0])
data_preprocessed.set_index('SK_ID_CURR', inplace=True)
print(f"dimension des données : {data_preprocessed.shape}\n")

#Train the model and save it to the folder model_and_data
X_train, X_test, y_train, y_test, feature_names = udf.data_reader(
        data_preprocessed, 100, True
    )

X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_test = X_test.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

X_train.to_csv(PROJECT_ROOT + '/model_and_data/X_train.csv')
X_test.to_csv(PROJECT_ROOT + '/model_and_data/X_test.csv')
y_train.to_csv(PROJECT_ROOT + '/model_and_data/y_train.csv')
y_test.to_csv(PROJECT_ROOT + '/model_and_data/y_test.csv')

print("Model training...")
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

randomforest_params = {
    'n_estimators': 882, 
    'max_depth': 20, 
    'min_samples_split': 6, 
    'min_samples_leaf': 10, 
    'bootstrap': True
    }
smote = SMOTE(random_state=1001, k_neighbors=17)
forest = RandomForestClassifier(**randomforest_params)
forest_smote = Pipeline([('smote', smote), ('model', forest)])

model_fited = forest_smote.fit(X_train, y_train)

import pickle
pickle.dump(model_fited, open(PROJECT_ROOT + '/model_and_data/model.pkl', 'wb'))

print("Model global feature importance...")
#get global feature importance
feature_importance = model_fited['model'].feature_importances_
f_importance_df = pd.DataFrame([X_train.columns.tolist(), feature_importance]).T
f_importance_df.rename(columns={0: 'Feature', 1: 'Importance'}, inplace=True)
f_importance_df.sort_values(by='Importance', ascending=True, inplace=True)

#save the feature importance to csv file in the folder model_and_data
f_importance_df.to_csv(PROJECT_ROOT + '/model_and_data/golbal_feature_importance.csv')

print("Run the Flask app...")
#run the app.py file with the run command
#run(['python', PROJECT_ROOT + '/src/app.py'], stdout=PIPE, stderr=PIPE, universal_newlines=True)

if __name__ == '__main__':
    pass

