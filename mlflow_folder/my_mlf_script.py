"""MLFLow training"""

import mlflow

def main():
    client = mlflow.MlflowClient()

    mlflow.log_param("threshold", 4)
    mlflow.log_param("verbosity", "INFO")

    mlflow.log_metric("ma_metrique", 666)
    mlflow.log_metric("TimeToCompute", 1)

    mlflow.log_artifact("my_artifact.csv")


if __name__ == '__main__':
    main()
