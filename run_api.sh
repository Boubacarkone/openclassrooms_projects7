#!/bin/bash
#Script bash pour exécuter l'API ou app.py

# créer un environnement virtuel s'il n'existe pas déjà sinon le mettre à jour et l'activer
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
    pip install -r requirements.txt
fi


# exécuter l'API
export FLASK_APP=app.py
export FLASK_ENV=production
flask run -h 0.0.0.0

echo "API is running"
