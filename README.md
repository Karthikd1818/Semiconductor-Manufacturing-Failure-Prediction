# Semiconductor Manufacturing Failure Prediction

## 📌 Project Overview

This project uses Machine Learning to predict whether a semiconductor manufacturing sample will **Pass** or **Fail** based on sensor measurements collected during the manufacturing process.

The project follows an end-to-end Data Science workflow:

**Data → Data Cleaning → Missing Value Handling → Feature Selection → Model Training → Threshold Tuning → Model Saving → FastAPI → Streamlit**

The trained machine learning model is deployed as a REST API using **FastAPI** and connected to an interactive **Streamlit** application for predictions.

---

## 🎯 Objective

The main objective is to identify potential manufacturing failures from sensor data.

The model predicts:

* **PASS** — Manufacturing sample is predicted to pass quality requirements.
* **FAIL** — Manufacturing sample is predicted to fail quality requirements.

This type of predictive system can help identify potentially defective manufacturing samples and support quality-control processes.

---

## 📊 Dataset

The project uses a semiconductor manufacturing dataset containing sensor measurements and a target variable.

### Target

| Target | Meaning                     |
| ------ | --------------------------- |
| `Pass` | Manufacturing sample passed |
| `Fail` | Manufacturing sample failed |

For machine learning:

```text
Pass → 0
Fail → 1
```

The dataset contains a large number of sensor features with missing values.

---

## 🔎 Data Preprocessing

The following preprocessing steps were performed:

### 1. Remove unnecessary columns

The `timestamp` column was removed because it was not used as a predictive feature.

### 2. Remove features with excessive missing values

Features with more than **90% missing values** were removed.

After this step, the final dataset contained **586 sensor features**.

### 3. Missing Value Handling

Missing values were handled using:

```python
SimpleImputer(strategy="median")
```

Median imputation was included directly inside the final machine learning pipeline.

### 4. Feature Selection

The top **50 features** were selected using:

```python
SelectKBest(score_func=f_classif, k=50)
```

### 5. Feature Scaling

Selected features were standardized using:

```python
StandardScaler()
```

---

## 🤖 Machine Learning Model

Several classification approaches were evaluated during model development.

The final production model is:

```text
Logistic Regression
```

with:

```python
class_weight="balanced"
```

This was used to account for the class imbalance in the target variable.

The final pipeline is:

```text
Median Imputation
        ↓
SelectKBest
        ↓
StandardScaler
        ↓
Balanced Logistic Regression
```

---

## 🎚️ Decision Threshold

Instead of using the default classification threshold of `0.5`, validation experiments were performed using different thresholds.

The selected production threshold is:

```text
0.7
```

The model predicts:

```text
FAIL → probability >= 0.7
PASS → probability < 0.7
```

This threshold is applied after obtaining the failure probability from the trained model.

---

## 💾 Model Saving

The complete preprocessing and model pipeline was saved using `joblib`.

The saved model contains:

```python
{
    "pipeline": final_pipeline,
    "threshold": 0.7,
    "feature_names": [...]
}
```

This ensures that the same preprocessing steps used during training are applied during prediction.

---

# 🚀 Deployment

The project contains two deployment components.

## 1. FastAPI Backend

FastAPI provides a REST API for making predictions.

### API endpoints

#### Health Check

```text
GET /health
```

Used to verify that the API and model are loaded correctly.

#### Prediction

```text
POST /predict
```

Accepts sensor values and returns the predicted result and failure probability.

Example response:

```json
{
  "status": "success",
  "threshold": 0.7,
  "results": [
    {
      "sample": 1,
      "prediction": "PASS",
      "fail_probability": 0.12
    }
  ]
}
```

---

## 2. Streamlit Frontend

The Streamlit application provides a simple user interface where users can upload a CSV file containing sensor measurements.

The application:

1. Accepts a CSV file.
2. Validates the required sensor features.
3. Keeps the features in the same order used during training.
4. Sends the data to the FastAPI backend.
5. Receives predictions.
6. Displays PASS/FAIL predictions and failure probabilities.

---

# 🏗️ Project Architecture

```text
                 CSV Sensor Data
                        │
                        ▼
                ┌───────────────┐
                │   Streamlit   │
                │   Frontend    │
                └───────┬───────┘
                        │
                        │ HTTP Request
                        ▼
                ┌───────────────┐
                │    FastAPI    │
                │    Backend    │
                └───────┬───────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Saved ML Pipeline  │
             │                     │
             │ Median Imputation   │
             │        ↓            │
             │ SelectKBest (50)    │
             │        ↓            │
             │ StandardScaler      │
             │        ↓            │
             │ Logistic Regression │
             └──────────┬──────────┘
                        │
                        ▼
                 PASS / FAIL
                        │
                        ▼
                Failure Probability
```

---

# 📁 Project Structure

```text
Semiconductor-Manufacturing-Failure-Prediction/
│
├── main.py
├── app.py
├── secom_quality_model.joblib
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File                         | Description                        |
| ---------------------------- | ---------------------------------- |
| `main.py`                    | FastAPI backend and prediction API |
| `app.py`                     | Streamlit frontend                 |
| `secom_quality_model.joblib` | Trained ML pipeline and metadata   |
| `requirements.txt`           | Python dependencies                |
| `README.md`                  | Project documentation              |
| `.gitignore`                 | Files excluded from Git            |

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Karthikd1818/Semiconductor-Manufacturing-Failure-Prediction.git
```

Navigate into the project:

```bash
cd Semiconductor-Manufacturing-Failure-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

## Start FastAPI

Open a terminal and run:

```bash
uvicorn main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Start Streamlit

Open another terminal:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

# 🧪 Prediction Workflow

The prediction workflow is:

```text
CSV Upload
    ↓
Feature Validation
    ↓
Feature Ordering
    ↓
FastAPI Request
    ↓
Median Imputation
    ↓
Feature Selection
    ↓
Scaling
    ↓
Logistic Regression
    ↓
Failure Probability
    ↓
Threshold = 0.7
    ↓
PASS / FAIL
```

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Streamlit
* Joblib
* REST API
* Machine Learning

---

# 💡 Key Learning Outcomes

Through this project, I worked on:

* Handling high-dimensional sensor data
* Missing-value analysis and imputation
* Feature selection
* Feature scaling
* Class imbalance
* Logistic Regression
* Model evaluation
* Decision threshold tuning
* Building an end-to-end ML pipeline
* Saving and loading ML models
* REST API development with FastAPI
* Building an interactive ML application with Streamlit
* Connecting a frontend application with an ML API

---

# 🔮 Future Improvements

Possible future improvements include:

* Deploying the FastAPI backend to a cloud platform
* Deploying the Streamlit application publicly
* Adding prediction history
* Adding model monitoring
* Adding probability visualizations
* Experimenting with additional classification algorithms
* Adding automated testing
* Implementing CI/CD

---

## 👨‍💻 Author

**Karthik D**

Aspiring Data Scientist

GitHub: https://github.com/Karthikd1818

LinkedIn: https://www.linkedin.com/in/karthik02052005/
