import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from mlflow.models.signature import infer_signature
from data_prep import load_raw_data, preprocess_and_split

def objective(trial):
    df = load_raw_data()
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    with mlflow.start_run(nested=True):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        })
        mlflow.log_metric("roc_auc", roc_auc)

    return roc_auc

def train_best_model(best_params):
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

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    mlflow.log_artifact("../models/preprocessor.pkl", artifact_path="preprocessor")

    
    joblib.dump(model, "../models/model.joblib")

    print(f"Best model trained. Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")
    print("Model saved to models/model.joblib")

def main():
    mlflow.set_experiment("stroke-prediction")

    with mlflow.start_run(run_name="optuna-hyperparameter-tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        print(f"Best trial params: {study.best_params}")
        print(f"Best trial ROC AUC: {study.best_value}")

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_roc_auc", study.best_value)

    with mlflow.start_run(run_name="best-model-training"):
        train_best_model(study.best_params)

if __name__ == "__main__":
    main()
