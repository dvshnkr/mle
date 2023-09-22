from flask import Flask, jsonify, request
from lambda_function import predict

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Creates the route using the imported endpoint function.
    """
    client = request.get_json()
    url = client["url"]

    return jsonify({"prediction": predict(url)})


if __name__ == "__main__":
    app.run(debug=True)
