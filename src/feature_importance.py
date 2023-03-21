import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from tqdm import tqdm
import re
from utils import udf
import os
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.inspection import permutation_importance
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
experiment_id = dict(mlflow.set_experiment("Model feature importances"))['experiment_id']

#database
data = pd.read_csv(Path(PROJECT_ROOT + '/data/data_pre_processed_final_v0.csv'), index_col=[0])
data.set_index('SK_ID_CURR', inplace=True)
print(f"dimension des données : {data.shape}\n")

X_train, X_test, y_train, y_test, feature_names = udf.data_reader(data, 50)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)

#cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=1001)
smote = SMOTE(random_state=1001, k_neighbors=17)

#Model list:
models = {}

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

scoring = {
    'f3_score': make_scorer(udf.f3_score, needs_proba=True),
    'accuracy': make_scorer(udf.custom_accuracy_score, needs_proba=True),
    'recall_score': make_scorer(udf.custom_recall_score, needs_proba=True),
    'preciion_score': make_scorer(udf.custom_precision_score, needs_proba=True)
}
i = 0
for model_name, model in tqdm(models.items()):
    i += 1
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=model_name):
    
        print(f"\nLe model {model_name} en traitement...\n")

        model_fited = model.fit(X_train, y_train)
        result = permutation_importance(
            model_fited, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )

        forest_importances_mean = pd.Series(result.importances_mean, index=feature_names[1:]).rename('mean')
        forest_importances_std = pd.Series(result.importances_std, index=feature_names[1:]).rename('std')

        feature_importance_df = pd.DataFrame([forest_importances_mean, forest_importances_std]).T[1:]
        feature_importance_df = feature_importance_df.sort_values('mean', ascending=False)

        plt.figure(figsize=(9, 8))
        sns.barplot(x='mean', y=feature_importance_df.index[:40], data=feature_importance_df[:40])
        sns.scatterplot(x='std', y=feature_importance_df.index[:40], data=feature_importance_df[:40], legend='auto')
        plt.title(f"{model_name} Feature importances using permutation")
        image_path = str(str(Path(PROJECT_ROOT)) + f"/data/plots/feature_importance_{model_name}.png")
        plt.savefig(image_path)
        plt.close()

        mlflow.log_artifact(image_path, artifact_path='plots')

        #Les variables les plus important selon le modèle
        important_features = feature_importance_df[feature_importance_df['mean'] > 0].index.to_list()
        print(f"Le nombre de variables importantes : {len(important_features)}")

        #important_features.insert(0, 'SK_ID_CURR')
        important_features.insert(0,'TARGET')
        data_filltered = data[important_features]
        data_filltered.to_csv(Path(PROJECT_ROOT + f"/data/data_preprocessed_filltered_v{i}.csv"))

        X_train, X_test, y_train, y_test, feature_names = udf.data_reader(data_filltered, 100)
        model_filltered_fited = model.fit(X_train, y_train)

        metrics, image_path = udf.evaluate_model_confusion_matrix(
                model_filltered_fited,
                model_name=model_name,
                X_test=X_test,
                y_test=y_test,
            )
        for metric_name, val in metrics.items():
            mlflow.log_metric(metric_name, val)

        mlflow.sklearn.log_model(
                sk_model=model_filltered_fited,
                artifact_path='models',
                registered_model_name=model_name
            )
        mlflow.log_artifact(image_path, artifact_path='plots')
