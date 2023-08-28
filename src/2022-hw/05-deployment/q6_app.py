# pylint: disable=R0801 (duplicate-code)

import pickle

from flask import Flask, jsonify, request

app = Flask(__name__)


def load_model():
    """
    Loads model to make predictions.
    """
    # loads DictVectorizer and model
    with open("./dv.bin", "rb") as f_in:
        dict_vzer = pickle.load(f_in)
    with open("./model2.bin", "rb") as f_in:
        model = pickle.load(f_in)
    return dict_vzer, model


def predict(client: dict) -> float:
    """
    Returns a prediction for the client.
    """
    dict_vzer, model = load_model()
    client_enc = dict_vzer.transform(client)  # transforms data
    # predict probably of client belonging to positive class
    y_hat = model.predict_proba(client_enc)[:, 1][0]
    return y_hat


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Creates an endpoint for users to send information.
    Returns the prediction made by the model.
    """
    client = request.get_json()  # receives JSON input with features
    prediction = predict(client)  # makes prediction
    return jsonify({"prediction": prediction})  # returns prediction in JSON format


if __name__ == "__main__":
    app.run(debug=True)
