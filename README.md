# OpenClassrooms_Project7
Développer un dashboard interactif

## Sujet du projet

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

## Data source

Les données téléchargées sur le lien donnée dans le consignes est : data/Projet+Mise+en+prod+-+home-credit-default-risk.zip
Son contenu dézipé contenant plusieurs fichier csv dans le folder data/tables.

## Organisation du git
Organisation en branche : une branche par fonctionnalité ;
- la branche data_preprocessing : pour la préparation des données (application du kenel kaggle, sélection de features ...)
- la branche modelisation : pour la construction des modèles (modèles sélection, optimisation...)
- la branche prediction_api :  pour le déploiement du modèle final via CI/CD de git action à chaque push sur cette branche.
- la branche Dashboard : pour l'implémentation et le déploiement du dashbord via une autre CI/CD de git action à chaque push sur cette branche aussi.

## Librairie utilisées dans le projet 
- La liste des librairie pour l'environement du projet se trouve dans les branche modélisation, prediction_api et Dashboard, dans le fichier nomé requirement.txt.
