#Data pre-processing
#Feature engineering : Utilisation de kernels Kaggle  + personnalisation
#Feacture sélection : corrélation + permutation importance du randomforestClassifer
#Sauvegarde des données prétraitées

import sys

#Correct the error no module named 'utils' when running the script main.py
sys.path.append('src/')
sys.path.append('src/utils/')


import pandas as pd
pd.options.mode.chained_assignment = None
from pathlib import Path
from tqdm import *
import re


from IPython.display import display_html #print de dataframes
from utils import lightgbm_with_simple_features as ksf #kaggle kernel
from utils import udf #Fonctions et méthodes
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
sns.set_theme(style="ticks")

import os
# directory reach
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(PROJECT_ROOT).parent

import time
from contextlib import contextmanager
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}mns".format(title, (time.time() - t0)/60))

def data_process(Tables_path:Path, n_step=1, data_rat=None, debug=False):
    #Recupération des données
    print("\nRecupération des données...")
    data_paths = []
    Datas = {}
    for root, dir, files in os.walk(Tables_path):
        
        for file in files:
            
            data_paths.append(os.path.join(root, file))

    for data_path in tqdm(data_paths):
        
        try:
            Datas[Path(data_path).name[:-4]+ '_df'] = pd.read_csv(data_path)
            
        except UnicodeDecodeError:
            Datas[Path(data_path).name[:-4]+'_df'] = pd.read_csv(data_path, encoding = "ISO-8859-1", index_col=[0])


    #Le nom des tables
    print("Les tables recupérées :\n")
    for dataset in Datas.keys():
        print(dataset)

    print("\nFeature ingineering start...", "\n")

    #Chargement des données prétraitées
    if debug:
        data_preproced = ksf.main(debug=True)
    else:
        data_preproced = ksf.main()

    del data_preproced['index']
    data_preproced = udf.reduce_mem_usage(data_preproced)

    print("\nData preprocessing : Sélection de features")

    # All
    #Corrélation entre variables
    SK_ID_CURR = data_preproced['SK_ID_CURR']
    corrs = data_preproced.corr()

    #Corrélation avec la variable cible target
    corr_with_target = pd.DataFrame(corrs['TARGET']).reset_index().round(2)
    corr_with_target.rename(columns={'index':'Variable', 'TARGET': 'corr'}, inplace=True)
    corr_with_target['corr'] = np.abs(corr_with_target['corr'])

    df = corr_with_target[corr_with_target['corr'].isna() | corr_with_target['corr'] == 0]
    features_not_corr_with_target = df['Variable'].to_list()
    print(f"Nombre de features non corrélées avec la target : {len(features_not_corr_with_target)}")

    #Corrélation entre variables prises deux à deux
    #Le seuil
    threshold = 0.8

    # stock de variables corrélées au delàs du seuil
    above_threshold_vars = {}

    # Pour chaque colonne, sauvegarder les variables qui sont au-dessus du seuil
    for col in corrs:
        above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])

    # Suivre les colonnes à supprimer et les colonnes déjà examinées
    cols_to_remove = []
    cols_seen = []
    cols_to_remove_pair = []

    # Itérer à travers les colonnes et les colonnes corrélées
    for key, value in above_threshold_vars.items():
        # Gardez les colonnes déjà examinées
        cols_seen.append(key)
        for x in value:
            if x == key:
                next
            else:
                # N'en retirez qu'un par paire
                if x not in cols_seen:
                    cols_to_remove.append(x)
                    cols_to_remove_pair.append(key)
                
    cols_to_remove = list(set(cols_to_remove))
    print('Nombre de colonnes à supprimer: ', len(cols_to_remove))

    print("Merge des variables à supprimer")
    features_to_remove = set(features_not_corr_with_target).union(set(cols_to_remove))
    features_to_remove = list(set(features_to_remove))
    print(f"Nombre de variable à supprimer : {len(features_to_remove)}", "\n")

    #Suppression des variables précédemment ciblées
    for col in tqdm(features_to_remove):
        
        try:
            del data_preproced[col]
        except:
            pass
    
    if debug:
        data_preproced['SK_ID_CURR'] = SK_ID_CURR
        data_preproced.to_csv(Path(str(PROJECT_ROOT) + '/data/data_preprocessed_final_debug_v0.csv'))
    else:
        data_preproced['SK_ID_CURR'] = SK_ID_CURR
        if data_rat is not None:
            percentage = data_rat
            nb_class_1 = int(len(data_preproced[data_preproced.TARGET == 1])*percentage/100)
            nb_class_0 = int(len(data_preproced[data_preproced.TARGET == 0])*percentage/100)
            nb_class_nan = int(len(data_preproced[data_preproced.TARGET.isna()])*percentage/100)

            data_preproced = pd.concat([data_preproced[data_preproced.TARGET == 1].sample(nb_class_1, random_state=42),
                                        data_preproced[data_preproced.TARGET == 0].sample(nb_class_0, random_state=42),
                                        data_preproced[data_preproced.TARGET.isna()].sample(nb_class_nan, random_state=42)]).reset_index(drop=True)
            data_preproced.to_csv(Path(str(PROJECT_ROOT) + '/data/data_pre_processed_final_v0.csv'))
            
        else:
            data_preproced.to_csv(Path(str(PROJECT_ROOT) + '/data/data_pre_processed_final_v0.csv'))

    print(f"Dimension des données après suppréssion des varaibles précédement ciblés : {data_preproced.shape}")
    print(f"Nombre de variables supprimées : {798 - data_preproced.shape[1]}", "\n")

    if n_step == 1:
        
        print("Feature sélection par feature permutation importance de randomforestClassier...\n")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.inspection import permutation_importance

        
        #Gestion du déséquilibre de TARGET
        datas = data_preproced[data_preproced.TARGET == 1]
        datas = pd.concat([datas, data_preproced[data_preproced.TARGET == 0].sample(len(datas), random_state=42)], ignore_index=True)
        print(f"Dimension des données sélectionnés pour cette étape : {datas.shape}")

        print("Separation des données")
        train_df = datas[datas['TARGET'].notnull()]
        test_df = datas[datas['TARGET'].isnull()]
        feature_names = train_df.columns.to_list()
        print("Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

        #Variable à normalisées
        var_to_norm = []
        for var in datas.columns:
            
            if len(datas[var].value_counts()) > 2:
                var_to_norm.append(var)
        print(f"Nombre de variable à normalisées : {len(var_to_norm)}")

        scaler = StandardScaler()
        train_df[var_to_norm] = scaler.fit_transform(train_df[var_to_norm])

        print("Imputation avec la valeur 0")
        train_df.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)
        print(f"Nombre de valeur manque après imputation : {train_df.isna().sum().sum()}")

        train_df = udf.remove_infite(train_df)
        test_df = udf.remove_infite(test_df)

        y = train_df['TARGET']
        del train_df['TARGET']
        del test_df['TARGET']

        X_train, X_test, y_train, y_test = train_test_split(train_df, y, stratify=y, random_state=42)
        print(f"Train Dimension: \nX_train : {X_train.shape} \nX_test : {X_test.shape} \ny_train : {y_train.shape} \ny_test : {y_test.shape}")

        forest = RandomForestClassifier(random_state=42)
        forest.fit(X_train, y_train)

        result = permutation_importance(
            forest, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        print("\nPermutation importance terminée\n")
        forest_importances_mean = pd.Series(result.importances_mean, index=feature_names[1:]).rename('mean')
        forest_importances_std = pd.Series(result.importances_std, index=feature_names[1:]).rename('std')
        feature_importance_df = pd.DataFrame([forest_importances_mean, forest_importances_std]).T[1:]
        feature_importance_df = feature_importance_df.sort_values('mean', ascending=False)

        print("\nTracé du Graphique de corrélation\n")
        plt.figure(figsize=(9, 8))
        sns.barplot(x='mean', y=feature_importance_df.index[:20], data=feature_importance_df[:20])
        sns.scatterplot(x='std', y=feature_importance_df.index[:20], data=feature_importance_df[:20], legend='auto')
        plt.title("Feature importances using permutation importance")
        if debug:
            plt.savefig(str(str(Path(PROJECT_ROOT)) + '/data/permutation_importance_debug.png'))
        else:
            plt.savefig(str(str(Path(PROJECT_ROOT)) + '/data/permutation_importance.png'))
        plt.close()
    
        #Les variables les plus important selon le modèle
        important_features = feature_importance_df[feature_importance_df['mean'] > 0].index.to_list()
        print(f"Le nombre de variables les plus importantes est : {len(important_features)}")

        print("Sauvegarde des données avec les features sélectionnées")
        important_features.insert(0,'TARGET')
        if debug:
            data_preproced = data_preproced[important_features]
            data_preproced['SK_ID_CURR'] = SK_ID_CURR
            data_preproced.to_csv(Path(str(PROJECT_ROOT) + '/data/data_preprocessed_final_debug.csv'))
        else:
            data_preproced = data_preproced[important_features]
            data_preproced['SK_ID_CURR'] = SK_ID_CURR
            data_preproced.to_csv(Path(str(PROJECT_ROOT) + '/data/data_pre_processed_final.csv'))

        print("\nFin des la preparation des données !")

if __name__ == "__main__":

    with timer("Data preprocessing run"):
        data_process(Path(str(PROJECT_ROOT) + '/data/tables'), debug=False)
    