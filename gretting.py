from flask import Flask, request, jsonify
from io import StringIO
import pandas as pd
import requests
from io import StringIO, BytesIO
import pickle

app = Flask(__name__)


def read_csv_from_azure(relatif_path:str):

    # URL SAP de l'objet blob
    if 'data' in relatif_path:
        sap_blob_url = "https://dashboardd.blob.core.windows.net/dashboarddata/" + relatif_path
    else:
        sap_blob_url = "https://dashboardd.blob.core.windows.net/dashboarddata/model_and_data/" + relatif_path

    # Récupération du fichier csv
    r = requests.get(sap_blob_url)

    # Création du dataframe
    df = pd.read_csv(StringIO(r.text), index_col=[0])
    return df


# load data to train the model and deploy it with flask
# data:
# for debug:
print("Loading data...")
# X_train = read_csv_from_azure("X_train.csv")
# print("X_train loaded", X_train.shape)
# X_test = read_csv_from_azure("X_test.csv")
# print("X_test loaded", X_test.shape)
# y_train = read_csv_from_azure("y_train.csv")
# print("y_train loaded", y_train.shape)
# y_test = read_csv_from_azure("y_test.csv")
# print("y_test loaded", y_test.shape)

test_df = read_csv_from_azure("test_df.csv")
print("test_df loaded", test_df.shape)

print("loading model...")
# Chargement d'un model depuis un fichier pickle depuis Azure
sap_blob_url = "https://dashboardd.blob.core.windows.net/dashboarddata/model_and_data/model.pkl"
r = requests.get(sap_blob_url)
model = pickle.load(BytesIO(r.content))
print("Model loaded")

print("Model trained")
@app.route('/predict/', methods=['GET'])
def respond():
    # Retrieve the name and first_name from the url parameter /getmsg/?name=
    client_id = request.args.get("client_id", None)
    feature_importance = request.args.get("return_local_feature_imp", None)

    # For debugging
    #print(f"Received: {name}")

    response = {}

    # Check if the user sent a name at all
    if not client_id:
        response["ERROR"] = "No client ID found. Please send a Client ID."
    # Check if the user entered a string
    elif isinstance(client_id, str):
        response["ERROR"] = "The client ID can't be string. Please send a number."
    else:

        # Get the predic_proba of the client_id
        proba = model.predict_proba(test_df[test_df.index == client_id].values[0])[0][1]
        response["proba"] = proba
        # Check if the user wants to see the local feature importance
        if feature_importance:
            response["local_feature_importance"] = "The feature importance not supported yet" #local_explainer(client_id, explainer).to_dict()
            
        else:
            pass
            #response["MESSAGE"] = f"Welcome {name}, {first_name} to our awesome API!"

    # Return the response in json format
    return jsonify(response)


@app.route('/post/', methods=['POST'])
def post_something():
    param = request.form.get('name')
    print(param)
    # You can add the test cases you made in the previous function, but in our case here you are just testing the POST functionality
    if param:
        return jsonify({
            "Message": f"Welcome {param} to our awesome API!",
            # Add this option to distinct the POST request
            "METHOD": "POST"
        })
    else:
        return jsonify({
            "ERROR": "No name found. Please send a name."
        })


@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Welcome to our medium-greeting-api!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
