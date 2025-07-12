# 🧠 Stroke Prediction - ML Project with MLOps

This project develops an end-to-end machine learning pipeline to predict the likelihood of a stroke in a patient using medical and demographic data. It follows MLOps best practices: preprocessing, training, hyperparameter tuning, experiment tracking, and model artifact logging.

---

## 🩺 Problem Description

Stroke is one of the leading causes of death and long-term disability worldwide. Early prediction of stroke risk can enable preventive care and save lives.

This project aims to build a predictive model using patient data such as:

- Age, gender, and BMI  
- Hypertension and heart disease  
- Smoking status, marital status  
- Residence type and work type  
- Average glucose level  

The target variable is `stroke` (binary: 1 = stroke occurred, 0 = no stroke).

---

## 📊 Dataset

- **Source**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Samples**: ~5,000 rows
- **Features**: Demographic + health indicators

The `id` column is dropped, and missing values in the `bmi` column are handled during preprocessing.

---

## ⚙️ Data Preprocessing

File: `src/data_prep.py`

We use a preprocessing pipeline based on `scikit-learn`:

- **Numerical features** (`age`, `avg_glucose_level`, `bmi`)  
  - Imputation: median  
  - Scaling: standard scaler  
- **Categorical features** (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`)  
  - One-hot encoding with unknown handling

Preprocessing is performed via a `ColumnTransformer`, saved using `joblib` to `models/preprocessor.pkl`. The data is split (80/20) using stratified sampling to preserve class distribution.

---

## 🧪 Model Training & Experiment Tracking

File: `src/train.py`

### ✅ Model Used

We use a **Random Forest Classifier**, which performs well on structured/tabular data and is robust to outliers and irrelevant features.

### 🔍 Hyperparameter Tuning

We use **Optuna**, an efficient hyperparameter optimization library that supports:

- **TPE (Tree-structured Parzen Estimator)**: A Bayesian optimization method that models the objective function and samples from a probability distribution to find better parameters quickly.

The following hyperparameters were optimized:

- `n_estimators`  
- `max_depth`  
- `min_samples_split`  
- `min_samples_leaf`  

Each trial's parameters and ROC AUC score are logged to MLflow automatically.

### 🧠 Final Model

After tuning, we retrain the best model on the full training set. Metrics logged:

- Accuracy  
- ROC AUC  
- Best hyperparameters  
- Input example + model signature  
- Preprocessor pipeline artifact  

---

## 🔧 Tools & Libraries

| Category               | Tool/Library             | Description                                            |
|------------------------|--------------------------|--------------------------------------------------------|
| Data handling          | `pandas`, `numpy`        | Load and manipulate data                               |
| Modeling               | `scikit-learn`           | Random Forest, preprocessing, metrics                  |
| Experiment tracking    | `MLflow`                 | Logs runs, parameters, metrics, artifacts              |
| Hyperparameter tuning  | `Optuna`                 | Efficient optimization using TPE algorithm             |
| Persistence            | `joblib`                 | Saving preprocessing pipeline                          |

---

## 📁 Project Structure

.
├── data/
│ └── raw/ # Raw dataset
├── models/
│ └── preprocessor.pkl # Saved transformer pipeline
├── src/
│ ├── data_prep.py # Data loading and preprocessing
│ └── train.py # Training, tuning, MLflow logging
├── README.md


---

## 📈 Results (Preliminary)

| Metric     | Value      |
|------------|------------|
| Accuracy   | *TBD*      |
| ROC AUC    | *TBD*      |

Will be updated once training is finalized.

---

## 🚧 Next Steps

- [ ] Add workflow orchestration (Mage or Prefect)
- [ ] Add model deployment (batch/REST)
- [ ] Set up model monitoring (e.g., Evidently)
- [ ] Add tests and CI/CD pipeline
- [ ] Use IaC (Terraform or similar)

---

## 📌 Notes

- MLflow provides a UI to compare experiment runs
- Preprocessing and model artifacts are reusable
- Project is modular and reproducible