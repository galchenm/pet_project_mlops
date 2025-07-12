# 🧠 Stroke Prediction - ML Project with MLOps

This project develops an end-to-end machine learning pipeline to predict the likelihood of a stroke in a patient using medical and demographic data. It follows MLOps best practices: preprocessing, training, hyperparameter tuning, experiment tracking, and model artifact logging.

---

## 🩺 Problem Description

Stroke is one of the leading causes of death and long-term disability worldwide. Early prediction of stroke risk can enable preventive care and save lives.

This project aims to build a predictive model using patient data such as:

* Age, gender, and BMI
* Hypertension and heart disease
* Smoking status, marital status
* Residence type and work type
* Average glucose level

The target variable is `stroke` (binary: 1 = stroke occurred, 0 = no stroke).

---

## 📊 Dataset

* **Source**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
* **Samples**: \~5,000 rows
* **Features**: Demographic + health indicators

The `id` column is dropped, and missing values in the `bmi` column are handled during preprocessing.

---

## ⚙️ Data Preprocessing

File: `src/data_prep.py`

We use a preprocessing pipeline based on `scikit-learn`:

* **Numerical features** (`age`, `avg_glucose_level`, `bmi`)

  * Imputation: median
  * Scaling: standard scaler
* **Categorical features** (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`)

  * One-hot encoding with unknown handling

Preprocessing is performed via a `ColumnTransformer`, saved using `joblib` to `models/preprocessor.pkl`. The data is split (80/20) using stratified sampling to preserve class distribution.

---

## 🧪 Model Training & Experiment Tracking

File: `src/train.py`

### ✅ Model Used

We use a **Random Forest Classifier**, which performs well on structured/tabular data and is robust to outliers and irrelevant features.

### 🔍 Hyperparameter Tuning

We use **Optuna**, an efficient hyperparameter optimization library that supports:

* **TPE (Tree-structured Parzen Estimator)**: A Bayesian optimization method that models the objective function and samples from a probability distribution to find better parameters quickly.

The following hyperparameters were optimized:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

Each trial's parameters and ROC AUC score are logged to MLflow automatically.

### 🧠 Final Model

After tuning, we retrain the best model on the full training set. Metrics logged:

* Accuracy
* ROC AUC
* Best hyperparameters
* Input example + model signature
* Preprocessor pipeline artifact

---

## 🔧 Tools & Libraries

| Category               | Tool/Library      | Description                                        |
| ---------------------- | ----------------- | -------------------------------------------------- |
| Data handling          | `pandas`, `numpy` | Load and manipulate data                           |
| Modeling               | `scikit-learn`    | Random Forest, preprocessing, metrics              |
| Experiment tracking    | `MLflow`          | Logs runs, parameters, metrics, artifacts          |
| Hyperparameter tuning  | `Optuna`          | Efficient optimization using TPE algorithm         |
| Persistence            | `joblib`          | Saving preprocessing pipeline                      |
| Workflow Orchestration | `Prefect`         | Managing and scheduling ML workflows and pipelines |

---

## ⚙️ Workflow Orchestration with Prefect

We use **Prefect** to orchestrate and manage our ML workflows, providing:

* Easy-to-define flows and tasks to structure pipeline steps
* Local Prefect server during development for flow execution and monitoring
* Scalable and maintainable workflow management

Prefect helps improve reliability and observability of the pipeline execution.

For more details, see the [Prefect Documentation](https://docs.prefect.io/).

---

## 🚀 Model Serving with FastAPI

We serve the trained stroke prediction model via a FastAPI application that exposes a REST endpoint for inference.

### How to run the server

1. Make sure the trained model and preprocessor files are saved in the `models/` directory as:

* `models/model.joblib`
* `models/preprocessor.pkl`

2. Run the FastAPI server:

```bash
uvicorn src.serve_model:app --reload
```

The server will start on:

```
http://127.0.0.1:8000
```

### API Endpoint

* **POST** `/predict`

  Accepts patient data and returns stroke risk probability and binary prediction.

### Request Body JSON schema

```json
{
  "age": 65,
  "avg_glucose_level": 160,
  "bmi": 26,
  "gender": "Female",
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "smoking_status": "never smoked"
}
```

### Example curl request

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "age": 65,
  "avg_glucose_level": 160,
  "bmi": 26,
  "gender": "Female",
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "smoking_status": "never smoked"
}'
```

### Example response

```json
{
  "stroke_probability":0.05808,
  "stroke_prediction":0
}
```

* `stroke_probability`: predicted probability of stroke occurrence
* `stroke_prediction`: binary prediction (1 = stroke predicted, 0 = no stroke)

---
## Docker Deployment

### What has been done

* A Docker image was created to containerize the FastAPI model serving application.
* The image includes the trained model (`model.joblib`) and the preprocessor (`preprocessor.pkl`), along with all required dependencies.
* The FastAPI app listens on port 8000 and exposes a `/predict` endpoint to receive patient data and return stroke risk predictions.

---

### How to run the Docker container

1. **Build the Docker image**
   Run this command in the project root (where the `Dockerfile` is located):

   ```bash
   docker build -t stroke-predictor:latest .
   ```

   This command builds a Docker image named `stroke-predictor`.

2. **Run the Docker container**

   ```bash
   docker run -p 8000:8000 stroke-predictor:latest
   ```

   This starts the container and maps port 8000 inside the container to port 8000 on your local machine.

3. **Verify the API is running**

   Open your browser or use a tool like `curl` to access the API at:

   ```
   http://localhost:8000/docs
   ```

   This will show the interactive FastAPI Swagger UI.

---

### How to test the API with `curl`

Send a POST request with sample patient data:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
  "age": 67,
  "avg_glucose_level": 105.92,
  "bmi": 36.6,
  "gender": "Female",
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "smoking_status": "formerly smoked"
}'
```

Expected response:

```json
{
  "stroke_probability": 0.06944068350051612,
  "stroke_prediction": 0
}
```
---
## 📁 Project Structure

```
.
├── data/
│   └── raw/               # Raw dataset
├── models/
│   └── preprocessor.pkl   # Saved transformer pipeline
├── src/
│   ├── data_prep.py       # Data loading and preprocessing
│   └── train.py           # Training, tuning, MLflow logging
│   └── pipeline.py        # Workflow Orchestration with Prefect
│   └── serve_model.py     # Model Serving with FastAPI
├── README.md
├── Dockerfile
├── requirements.txt
```

---

## 📈 Results 

| Metric   | Value |
| -------- | ----- |
| Accuracy | 0.951 |
| ROC AUC  | 0.843 |


