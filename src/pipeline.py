# src/pipeline.py
import mlflow
import optuna
import pandas as pd
from prefect import task, flow, get_run_logger
from data_prep import load_raw_data, preprocess_and_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from mlflow.models.signature import infer_signature

@task
def tune_model(n_trials: int = 20):
    def objective(trial):
        df = load_raw_data()
        X_train, X_test, y_train, y_test = preprocess_and_split(df)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        with mlflow.start_run(nested=True):
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            mlflow.log_params(params)
            mlflow.log_metric("roc_auc", roc_auc)
        return roc_auc, params

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t)[0], n_trials=n_trials)
    best = study.best_trial
    return {"best_score": best.value, "best_params": best.params}

@task
def train_and_log(best_params: dict):
    logger = get_run_logger()
    df = load_raw_data()
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc_auc)

    input_example = pd.DataFrame(X_test).head(5)
    signature = infer_signature(input_example, model.predict(input_example))
    mlflow.sklearn.log_model(model, artifact_path="model",
                             signature=signature, input_example=input_example)
    mlflow.log_artifact("../models/preprocessor.pkl", artifact_path="preprocessor")

    logger.info(f"Model trained: accuracy={acc:.4f}, roc_auc={roc_auc:.4f}")
    return acc, roc_auc

@flow
def stroke_pipeline(n_trials: int = 20):
    mlflow.set_experiment("stroke-prediction")
    tuning = tune_model(n_trials)
    best_params = tuning["best_params"]
    score = tuning["best_score"]
    acc, roc_auc = train_and_log(best_params)
    return {"tuning_score": score, "test_accuracy": acc, "test_roc_auc": roc_auc}

if __name__ == "__main__":
    result = stroke_pipeline()
    print(result)
