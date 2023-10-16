# pylint: disable = R0801

import pickle

from flask import jsonify, request


def load_model(filepath: str):
    """
    Accepts path/to/model.
    Loads and returns model.
    """
    # loads and returns model
    with open(filepath, "rb") as f_in:
        model = pickle.load(f_in)
    return model


def predict(client: dict, modelname: str) -> float:
    """
    Returns a prediction for the client.
    """
    dict_vzer = load_model("dv.bin")
    model = load_model(modelname)
    client_enc = dict_vzer.transform(client)  # transforms data
    # predict probably of client belonging to positive class
    y_hat = model.predict_proba(client_enc)[:, 1][0]
    return y_hat


def predict_endpoint(modelname: str):
    """
    Creates an endpoint for users to send information.
    Returns the prediction made by the model.
    """
    client = request.get_json()  # receives JSON input with features
    prediction = predict(client, modelname)  # makes prediction
    return jsonify({"prediction": prediction})  # returns prediction in JSON format


if __name__ == "__main__":
    print("Helper functions for the prediction endpoint...")
