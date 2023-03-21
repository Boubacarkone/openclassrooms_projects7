#Les méthodes et fonctions utilisées dans le projet, défini pas moi ou provenant d’autre kernel kaggle.
#Imports:
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import RocCurveDisplay, auc, roc_curve, fbeta_score, ConfusionMatrixDisplay, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_theme(style="ticks")
import re
import warnings
import os
from pathlib import Path
import optuna
warnings.simplefilter(action='ignore', category=FutureWarning)

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = PROJECT_ROOT.parent.parent


#Kaggle codes
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def remove_infite(df):
    return df[np.isfinite(df).all(1)]


#Hyperparameters tuning functions
# Define the helper function so that it can be reused
def tune(objective, n_trials_num):
    study = optuna.create_study(
        direction="maximize", 
        study_name = 'Hyperparameters optimization',
        pruner = optuna.pruners.HyperbandPruner()
        )
    study.optimize(objective, n_trials=n_trials_num)

    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}\n")
    print(f"Optimized parameters: {params}\n")
    return params

#Customize scoring
proba_values = []
def custom_metric(
        y_true, 
        y_pred_proba, 
        proba_values=proba_values
        ):
    """Cherche le seuil de probabilité qui minimise 10FN + FP et maximise le score métier
    et return les labels predits"""
    
    count = 0
    for threshold in np.arange(0.2, 0.9, 0.15):
        label_pred_threshold = np.where(y_pred_proba >= threshold, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true, label_pred_threshold).ravel()
        f3_score = fbeta_score(y_true, label_pred_threshold, beta=np.sqrt(10))
        #acc = accuracy_score(y_true, label_pred_threshold)
        #score = mean([acc, f3_score])
        
        if count >= 10*fn + fp - f3_score or count == 0:
            count = 10*fn + fp - f3_score
            
            threshold_f3 = threshold
            proba_values.append(threshold)
            #print(count)
        #else:
            #print("not valid", count, threshold, threshold_f3)
    label_pred_threshold = np.where(y_pred_proba >= threshold_f3, 1, 0)
    return label_pred_threshold, threshold_f3

def f3_score(y_true, y_pred_proba):
    label_pred_threshold = custom_metric(y_true=y_true, y_pred_proba=y_pred_proba)[0]
    f3_score = fbeta_score(y_true, label_pred_threshold, beta=np.sqrt(10))
    return f3_score


def custom_accuracy_score(y_true, y_pred_proba):
    label_pred_threshold = custom_metric(y_true=y_true, y_pred_proba=y_pred_proba)[0]
    acc = accuracy_score(y_true, label_pred_threshold)
    return acc

def custom_recall_score(y_true, y_pred_proba):
    label_pred_threshold = custom_metric(y_true=y_true, y_pred_proba=y_pred_proba)[0]
    recall_scores = recall_score(y_true, label_pred_threshold)
    return recall_scores

def custom_precision_score(y_true, y_pred_proba):
    label_pred_threshold = custom_metric(y_true=y_true, y_pred_proba=y_pred_proba)[0]
    precision_scores = precision_score(y_true, label_pred_threshold)
    return precision_scores


#model evaluation function
def evaluate_model_confusion_matrix(model, model_name, X_test, y_test):
    """prend en paramètre le modèle, les données de test et un seuil de classification
    trace la matrice de confusion et retunr les metrics"""
    
    # prediction des labels pour un seuil de classification
    pred_proba = model.predict_proba(X_test)[:,1]
    threshold = custom_metric(y_test, pred_proba)[1]
    
    label_pred_threshold = np.where(pred_proba >= threshold, 1, 0)
    scores = {
        'f3_score': fbeta_score(y_test, label_pred_threshold, beta=np.sqrt(10)),
        'accuracy_score': accuracy_score(y_test, label_pred_threshold),
        'recall_score': recall_score(y_test, label_pred_threshold),
        'precision_score': precision_score(y_test, label_pred_threshold),
        'threshold_proba': threshold
    }
    
    #Evaluation du modèle avec le seuil de classification précédent
    conf_mat = confusion_matrix(y_test, label_pred_threshold, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='.2f')
    plt.title('Confusion matrix', fontsize=16, fontweight='bold')
    image_path = str(str(Path(PROJECT_ROOT)) + f"/data/plots/{model_name}.png")
    plt.savefig(image_path)
    plt.close()
    return scores, image_path

def data_reader(data, percentage:int, save_test_df:bool=False):

    nb_class_1 = int(len(data[data.TARGET == 1])*percentage/100)
    nb_class_0 = int(len(data[data.TARGET == 0])*percentage/100)
    print(f"{percentage}% de la classe 0 : {nb_class_0}\n{percentage}% de la classe 1 : {nb_class_1}\n")

    
    df = pd.concat([
            data[data.TARGET == 0].sample(nb_class_0, random_state=42),
            data[data.TARGET == 1].sample(nb_class_1, random_state=42)
    ], ignore_index=True
    )

    # Separation des données
    train_df = df[df['TARGET'].notnull()]
    test_df = data[data['TARGET'].isnull()]
    print("Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    #cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=1001)

    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
    

    train_df = remove_infite(train_df)
    test_df = remove_infite(test_df)

    feature_names = train_df.columns.to_list()
    #Feature à normaliser
    var_to_norm = []
    for var in train_df.columns:
        
        if len(train_df[var].value_counts().values.tolist()) > 2:
            var_to_norm.append(var)
            
    scaler = StandardScaler()
    train_df[var_to_norm] = scaler.fit_transform(train_df[var_to_norm])
    test_df[var_to_norm] = scaler.transform(test_df[var_to_norm])
    print("Nombre de variables normalisés : {}".format(len(var_to_norm)))

    y = train_df['TARGET']
    del train_df['TARGET']
    del test_df['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(train_df, y, stratify=y, random_state=42)
    print(f"Train Dimension: \nX_train : {X_train.shape} \nX_test : {X_test.shape} \ny_train : {y_train.shape} \ny_test : {y_test.shape}")

    X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    X_test = X_test.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    test_df = test_df.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    if save_test_df:
        test_df.to_csv(str(str(Path(PROJECT_ROOT)) + f"/model_and_data/test_df.csv"))
        X_test.to_csv(str(str(Path(PROJECT_ROOT)) + f"/model_and_data/X_validation.csv"))
        y_test.to_csv(str(str(Path(PROJECT_ROOT)) + f"/model_and_data/y_validation.csv"))
    

    return X_train, X_test, y_train, y_test, feature_names
