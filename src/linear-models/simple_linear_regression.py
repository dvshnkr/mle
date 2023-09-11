from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def get_random_features(
    n_records: int = 100,
    n_features: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns first principal component of randomly generated features (X),
    along with outcome (y).
    """

    # initialises random number generator from numpy
    rng = np.random.default_rng(seed)

    X = rng.random(size=(n_records, n_features))
    pca = PCA(n_components=1, random_state=seed)
    X = pca.fit_transform(X)  # reduces X to one feature
    y = rng.random(size=(n_records,))

    return X, y


def plot_linear_regression(
    X: np.ndarray, y: np.ndarray, y_pred: np.ndarray = None
) -> None:
    """
    Plots a scatter plot using X and y.
    Also plots a linear regression line using model predictions for X.
    """

    _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:, 0], y)  # plots actual data as scatter plot

    if isinstance(y_pred, np.ndarray):
        # plots red line fitted on data
        ax.plot(X, y_pred, "r")

    ax.set_xlabel("Feature (X)")
    ax.set_ylabel("Outcome (y)")
    ax.set_title("Simple Linear Regression")
    plt.show()


def main() -> None:
    """
    Main function.
    """
    X, y = get_random_features()

    # trains a simple linear regression model
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    plot_linear_regression(X, y, y_pred)


if __name__ == "__main__":
    main()
