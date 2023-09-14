from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def plotbox(
    X: pd.core.frame.DataFrame,
    y: pd.core.series.Series,
    features: str | List[str] = "proline",
) -> None:
    """
    Plots a box plot of given feature, by Wine class.
    """

    sns.set_theme()

    if isinstance(features, str):
        _, ax = plt.subplots(figsize=(6, 6))
        sns.boxplot(x=y, y=X[features])
        ax.set_title(f"Boxplot of {features} by Wine Class")
        plt.show()

    if isinstance(features, List):
        if len(features) == 4:
            fig, ax = plt.subplots(2, 2, figsize=(9, 9), sharex=False, sharey=False)
            for i, a in enumerate(ax.flatten()):
                feat = features[i]
                sns.boxplot(x=y, y=X[feat], ax=a)
                # a.set_ylabel('')
                a.set_xlabel("")
                # a.set_title(f'{feat}')
            fig.suptitle("Boxplot of features by Wine Class")
            plt.show()


def train_logistic(
    X: pd.core.frame.DataFrame, y: pd.core.series.Series, C: float = 1.0
) -> float:
    """
    Trains logistic regression model on data.
    Returns accuracy score.
    """

    logistic = LogisticRegression(C=C)
    logistic.fit(X, y)

    y_pred = logistic.predict(X)
    acc = accuracy_score(y, y_pred)

    return acc


def train_logistic_cv(
    X: pd.core.frame.DataFrame, y: pd.core.series.Series, cv: int = 5
) -> float:
    """
    Trains logistic regression model on data.
    Returns accuracy score.
    """

    logistic = LogisticRegressionCV(cv=cv)
    logistic.fit(X, y)

    acc = logistic.score(X, y)

    return acc


def main():
    """
    Main function.
    """

    X, y = load_wine(as_frame=True, return_X_y=True)

    ### EDA ###
    # features = ["color_intensity", "alcohol", "magnesium", "flavanoids"]
    # plotbox(X, y, features)

    ### Pre-processing ###
    X = StandardScaler().fit_transform(X)

    ### Modeling without CV ###
    # acc = train_logistic(X, y, C=0.1)

    ### Modeling with CV ###
    acc = train_logistic_cv(X, y, cv=10)
    print(f"Accuracy score: {acc}")


if __name__ == "__main__":
    main()
