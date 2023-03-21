#Take model get it scores with cross validation and then send metrics, model to mlflow.
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from tqdm import tqdm
import re
from utils import udf
import os

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import RocCurveDisplay, auc, roc_curve, fbeta_score, ConfusionMatrixDisplay, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier

import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = str(PROJECT_ROOT.parent)

mlflow.set_tracking_uri("file://" + PROJECT_ROOT + "/mlruns")
experiment_id = dict(mlflow.set_experiment("Model selection (1)"))['experiment_id']
experiment_id

#database
data = pd.read_csv(Path(PROJECT_ROOT + '/data/data_pre_processed_final_v0.csv'), index_col=[0])
data.set_index('SK_ID_CURR', inplace=True)
print(f"dimension des données : {data.shape}\n")

X_train, X_test, y_train, y_test, feature_names = udf.data_reader(data, 100, save_test_df=True)
print(f"Train Dimension: \nX_train : {X_train.shape} \nX_test : {X_test.shape} \ny_train : {y_train.shape} \ny_test : {y_test.shape}")

X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_test = X_test.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)
smote = SMOTE(random_state=1001, k_neighbors=17)

#Model list:
models = {}

#KNneighborsclassifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(10)
knn_smote = Pipeline([('smote', smote), ('model', knn)])
models['KNeighborsClassifier'] = knn_smote

#LogisticRegression
from sklearn.linear_model import LogisticRegression

logistic_params = {
    'tol': 0.0002329433780056682, 
    'C': 6.681553108313681e-08, 
    'penalty': 'l2',
    'random_state': 42,
    'n_jobs': -1
    }
logistic_regression = LogisticRegression(**logistic_params)
logistic_regression_smote = Pipeline([('smote', smote), ('model', logistic_regression)])
models["LogisticRegression"] = logistic_regression_smote

#LGBM Classifier
from lightgbm import LGBMClassifier

lgdm_params = {
    'n_estimators': 1379,
    'learning_rate': 0.0305180705872894,
    'num_leaves': 1340,
    'max_depth': 11,
    'min_data_in_leaf': 200,
    'max_bin': 274,
    'lambda_l1': 100,
    'lambda_l2': 80,
    'min_gain_to_split': 5.6169036486489725,
    'bagging_fraction': 0.30000000000000004,
    'bagging_freq': 1,
    'feature_fraction': 0.30000000000000004,
    'random_state': 42,
    'n_jobs': -1
 }
lgbm = LGBMClassifier(**lgdm_params)
lgbm_smote = Pipeline([('smote', smote), ('model', lgbm)])

models["LGBMClassifier"] = lgbm_smote

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

randomforest_params = {
    'n_estimators': 882, 
    'max_depth': 20, 
    'min_samples_split': 6, 
    'min_samples_leaf': 10, 
    'bootstrap': True
    }

forest = RandomForestClassifier(**randomforest_params)
forest_smote = Pipeline([('smote', smote), ('model', forest)])
models["RandomForestClassifier"] = forest_smote

#Staking models logistic_regression + RandomForest
estimators_0 = [
    ('LogisticRegression', logistic_regression)
]
classifier_0 = StackingClassifier(
    estimators=estimators_0,
    n_jobs=-1,
    final_estimator=forest,
    stack_method='predict_proba',
    cv=cv,
    passthrough=True
)

clf_smote = Pipeline([('smote', smote), ('model', classifier_0)])

models['Stacked_models'] = clf_smote

#Stacked models optimzed together
logisticregression_params = {
    'tol': 0.0002168333834926689,
     'C': 0.00012720838557589821,
     'penalty': 'l2'
}

logisticregression = LogisticRegression(random_state=42, n_jobs=-1, **logisticregression_params)

randomforest_params = {
     'n_estimators': 311,
     'max_depth': 80,
     'min_samples_split': 9,
     'min_samples_leaf': 8,
     'bootstrap': True
}
randomforest = RandomForestClassifier(random_state=42, n_jobs=-1, **randomforest_params)

estimators = [
    ('logisticregression', logisticregression),
]

classifier = StackingClassifier(
    estimators=estimators,
    n_jobs=-1,
    final_estimator=randomforest,
    stack_method='predict_proba',
    cv=cv,
    passthrough=True
)
stacked_model_optimized = Pipeline([('smote', smote), ('model', classifier)])

models['Stacked_models_optimized'] = stacked_model_optimized



scoring = {
    'f3_score': make_scorer(udf.f3_score, needs_proba=True),
    'accuracy': make_scorer(udf.custom_accuracy_score, needs_proba=True),
    'recall_score': make_scorer(udf.custom_recall_score, needs_proba=True),
    'preciion_score': make_scorer(udf.custom_precision_score, needs_proba=True)
}

for model_name, model in tqdm(models.items()):

    
    with mlflow.start_run(experiment_id=experiment_id, run_name=model_name):
    
        print(f"\nLe model {model_name} en traitement...\n")
        """
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=scoring,
            #return_train_score=True,
            n_jobs=-1,
        )
        #print(scores)
        #print(scores.keys())
        
        for score_name, score in scores.items():
            #print(score_name, np.mean(score))
            mlflow.log_metric(score_name + '_mean', score.mean())
        """
        model_fited = model.fit(X_train, y_train)
        metrics, image_path = udf.evaluate_model_confusion_matrix(
                model_fited,
                model_name=model_name,
                X_test=X_test,
                y_test=y_test,
            )
        for metric_name, val in metrics.items():
            mlflow.log_metric(metric_name, val)

        mlflow.sklearn.log_model(
                sk_model=model_fited,
                artifact_path='models',
                registered_model_name=model_name
            )
        if model_name not in  ['Stacked_models_optimized', 'Stacked_models']:
            params = model_fited.get_params()
            for param_name, val in params.items():
                mlflow.log_param(param_name, val)
            
        mlflow.log_artifact(image_path, artifact_path='plots')
