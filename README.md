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

#### 📦 Installation

> *(Insert this near the top, right before or after the "Dataset" or "Tools & Libraries" section)*

### 📦 Installation

1. **Create a virtual environment**

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

2. **Install dependencies**

```bash
make install
```

3. **Set up pre-commit hooks**

```bash
pre-commit install
```

Now you're ready to run the pipeline, serve the model, or test the API.

---

#### 🧪 Testing, Code Quality, and Automation

> *(Append this at the end of the section that includes pre-commit and CI)*

### 🗂️ `.gitignore` Setup

The project uses a `.gitignore` file to prevent committing unnecessary or sensitive files such as:

* Python cache files (`__pycache__/`, `.pyc`)
* Virtual environment directories
* `models/` directory with trained artifacts
* `.env` or local config files (if any)

If you’ve added `.gitignore` **after** your initial commits, apply it retroactively like this:

```bash
git rm -r --cached .
git add .
git commit -m "Apply .gitignore rules to existing files"
```

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
## 📊 Monitoring with Prometheus and Grafana

This project includes basic monitoring for the FastAPI model server using **Prometheus** to collect metrics and **Grafana** to visualize them.

---

### ✅ Metrics Tracked

* `request_count_total`: total number of requests received
* `request_latency_seconds`: histogram of request durations

These metrics are exposed from the FastAPI app on the `/metrics` endpoint.

---

### ⚙️ Setup Instructions

We use **Docker containers** for Prometheus and Grafana, while the **FastAPI server runs on the host machine** using the `bridge` network.

---

### 1. Start FastAPI Server

Run the FastAPI server locally on port `8000`:

```bash
uvicorn src.serve_model:app --host 0.0.0.0 --port 8000 --reload
```

This serves the API (including `/metrics`) at:

```
http://localhost:8000
```

---

### 2. Prometheus Configuration

#### prometheus.yml

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'stroke-predictor'
    static_configs:
      - targets: ['host.docker.internal:8000']  # Use this on Mac/Windows

# On Linux, use your host's actual IP address instead:
# targets: ['172.17.0.1:8000']  # Replace with output of `ip addr show docker0`
```

#### Run Prometheus Container

If a container named `prometheus` already exists, remove it:

```bash
docker rm -f prometheus
```

Then start Prometheus:

```bash
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

---

### 3. Run Grafana Container

```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

* Access Grafana at: [http://localhost:3000](http://localhost:3000)
* Default login: `admin` / `admin`

---

### 4. Add Prometheus Data Source to Grafana

1. Go to **Grafana → Settings → Data Sources → Add Data Source**

2. Select **Prometheus**

3. Set the URL to:

   ```
   http://host.docker.internal:9090
   ```

   > On **Linux**, use your host IP (e.g. `http://172.17.0.1:9090`)

4. Click **Save & Test**

---

### 🧪 Test It Out

Send a few POST requests to `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "gender": "Male",
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "smoking_status": "formerly smoked"
  }'
```

---

### 📈 View Metrics in Grafana

Go to **Explore** and try these queries:

* `request_count_total`
* `rate(request_count_total[1m])`
* `histogram_quantile(0.95, rate(request_latency_seconds_bucket[1m]))`

---

## Testing, Code Quality, and Automation

### Pytest

We use **pytest** for automated testing. Tests are located in the `tests/` directory.
To run all tests:

```bash
pytest tests
```

This ensures code correctness and helps catch bugs early.

---

### Makefile

A `Makefile` is provided for common project tasks. Use the following commands:

| Command             | Description                                    |
| ------------------- | ---------------------------------------------- |
| `make install`      | Installs required Python packages.             |
| `make lint`         | Runs `flake8` linter on source code and tests. |
| `make format`       | Formats code using `black`.                    |
| `make test`         | Runs all tests with `pytest`.                  |
| `make serve`        | Starts the FastAPI model server.               |
| `make docker-build` | Builds the Docker image.                       |
| `make docker-run`   | Runs the Docker container.                     |

This standardizes workflows and simplifies common operations.

---

### Pre-commit Hooks

We use **pre-commit** hooks to maintain code quality by automatically running linters and formatters before every commit. This helps catch and fix style issues early.

To set up the hooks locally:

```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically on staged files when you commit. To run them on all files once, use:

```bash
pre-commit run --all-files
```
---
### Continuous Integration (CI)

We use **GitHub Actions** to automate linting and testing on every push and pull request.  
The workflow installs dependencies, runs `flake8` to check code style, and executes `pytest` for automated tests.

This ensures code quality and helps prevent regressions by validating code before merging.

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


