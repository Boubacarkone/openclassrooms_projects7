# OpenClassrooms_Project7
Développer un dashboard interactif

## Sujet du projet

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

## Data source

Les données téléchargées sur le lien donnée dans le consignes est : data/Projet+Mise+en+prod+-+home-credit-default-risk.zip
Son contenu dézipé contenant plusieurs fichier csv dans le folder data/tables.

## Branche prediction_api

- Deploier le modèle sur azure:
    - Recupère le code de déploiement sur une VM azure
    - Crée un environnement virtuel
    - recupère le modèle sur azure stokage et déploie le modèle