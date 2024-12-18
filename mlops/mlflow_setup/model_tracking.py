import mlflow
import mlflow.pytorch

def log_model(model, model_name="gpt-neo"):
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.pytorch.log_model(model, "model")