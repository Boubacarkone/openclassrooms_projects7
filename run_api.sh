#!/bin/bash
#Script bash pour exécuter l'API ou app.py

#activer l'environnement virtuel

source venv/bin/activate

# exécuter l'API
export FLASK_APP=app.py
export FLASK_ENV=production
#flask run -h 0.0.0.0

# exécuter l'API avec gunicorn
gunicorn -w 2 --threads 2 --max-requests 1000 --bind 0.0.0.0:5000 app:APP

echo "API is running"
