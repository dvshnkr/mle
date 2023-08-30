from endpoint_helper import predict_endpoint
from flask import Flask

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Creates the route using the imported endpoint function.
    """
    return predict_endpoint("model2.bin")


if __name__ == "__main__":
    app.run(debug=True)
