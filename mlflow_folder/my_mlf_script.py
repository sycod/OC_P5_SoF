"""MLFLow training"""

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


def main():
    mlflow.autolog()

    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    # MLflow triggers logging automatically upon model fitting
    rf.fit(X_train, y_train)



    # client = mlflow.MlflowClient()

    # mlflow.log_param("threshold", 4)
    # mlflow.log_param("verbosity", "INFO")

    # mlflow.log_metric("ma_metrique", 666)
    # mlflow.log_metric("TimeToCompute", 1)

    # mlflow.log_artifact("my_artifact.csv")


if __name__ == '__main__':
    main()
