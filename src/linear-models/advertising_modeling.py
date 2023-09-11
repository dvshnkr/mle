from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def plot_lin_reg(
    X: np.ndarray, y: np.ndarray, space: np.ndarray = None, y_pred: np.ndarray = None
) -> None:
    """Plots linear regression model predictions."""

    sns.set_theme()

    _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:, 0], y)  # plots actual data as scatter plot

    if isinstance(y_pred, np.ndarray):
        # plots red line fitted on data
        ax.plot(space, y_pred, "r")

    ax.set_xlabel("TV Advertising")
    ax.set_ylabel("Sales")
    ax.set_title("TV Advertising vs Sales")
    plt.show()


def linear_regression(
    data: pd.core.frame.DataFrame,
    features: str | List[str] = "TV",
    target: str = "Sales",
) -> None:
    """
    Train a simple linear regression using a single feature or a list of features.
    """

    # creates X and y from data
    if isinstance(features, str):
        X = data[features].values.reshape(-1, 1)
    else:
        X = data[features].values
    y = data[target].values

    # scales X
    X = StandardScaler().fit_transform(X)

    # fits linear model to data
    model = Lasso(alpha=1).fit(X, y)

    # prints model co-efficient and intercept
    print(f"Model intercept: {model.intercept_}")
    print(f"Model co-efficient: {model.coef_}")

    # makes predictions on training data
    y_pred = model.predict(X)

    # computes root mean squared error
    rmse = mean_squared_error(y, y_pred, squared=False)

    print(f"RMSE: {rmse}")

    if isinstance(features, str):
        # defines space for prediction
        space = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

        # makes predictions
        predictions = model.predict(space)

        # plots
        plot_lin_reg(X, y, space, predictions)


def poly_regression(
    data: pd.core.frame.DataFrame,
    features: List[str],
    target: str = "Sales",
) -> None:
    """
    Train polynomial regression.
    """

    poly = PolynomialFeatures(2)
    X = data[features]
    y = data[target]

    # creates polynomial features
    X = poly.fit_transform(X)

    # fits linear model
    model = LinearRegression(fit_intercept=False).fit(X, y)

    # prints model parameters
    print(f"Model intercept: {model.intercept_}")
    print(f"Model co-efficient: {model.coef_}")
    print()

    # makes predictions on training data
    y_pred = model.predict(X)

    # computes and prints RMSE
    rmse = mean_squared_error(y, y_pred, squared=False)
    print(f"RMSE: {rmse}")


def main():
    """Main function."""

    adv_data = pd.read_csv("data/Advertising.csv")

    features = ["TV", "Radio", "Newspaper"]
    # target = ['Sales']

    # linear_regression(adv_data, features=features)
    poly_regression(adv_data, features=features)


if __name__ == "__main__":
    main()
