# 🩺 Diabetes Risk Predictor

> A machine learning web application that estimates a user's diabetes risk from **15 health indicators**.
> Built with an **XGBoost** classifier trained on the **CDC BRFSS** survey dataset, a **Flask** REST API backend,
> **SHAP** for prediction explainability, and a two-page **HTML/CSS/JS** frontend — no framework required.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Dataset](#-dataset)
- [Frontend Pages](#-frontend-pages)
- [Input Features & Validation Rules](#-input-features--validation-rules)
- [Risk Levels](#-risk-levels)
- [API Reference](#-api-reference)
- [Setup & Installation](#-setup--installation)
- [Running the App](#-running-the-app)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

The user fills in a health questionnaire in the browser across four organized cards. The frontend sends
the answers to the Flask backend, which validates the inputs, runs them through the trained XGBoost model,
computes **SHAP explainability values** to show which factors drove the prediction, and returns a
**risk percentage (0–100%)** with a human-readable risk level.

The app is split across two pages: `index.html` collects the inputs, and `results.html` displays
the prediction with an **animated score counter**, **color-coded progress meter**, a full summary of
submitted health indicators, and **per-feature SHAP contribution breakdown**. Results are passed
between pages using `sessionStorage`.

---

## 📁 Project Structure

```
Diabetes-Prediction-AI/
│
├── diabetes_risk_model.py        # ML core: data loading, training, validation, SHAP, inference
├── app.py                        # Flask REST API (3 endpoints)
├── index.html                    # Page 1 — health questionnaire form
├── results.html                  # Page 2 — animated risk result, SHAP breakdown & input summary
├── requirements.txt              # Python dependencies
│
├── diabetes_risk_model.pkl       # Auto-generated on first run — cached trained model
└── Diabetes Health Indicators Dataset export 2026-02-27 17-12-07.csv
```

> **Note:** The `.pkl` cache file is created automatically the first time you run the app.
> You do **not** need to train the model manually.

---

## ⚙️ How It Works

```
Browser (index.html)
        │
        │  User fills form → clicks "Assess My Risk"
        │
        │  POST /predict  { feature: value, ... }
        ▼
Flask backend (app.py)
        │
        ├── Type coercion  (strings → int/float)
        ├── validate_inputs()  (bounds & type checking)
        │
        ├── predict_risk()
        │       ├── StandardScaler.transform()
        │       ├── XGBClassifier.predict_proba()
        │       └── SHAP TreeExplainer → per-feature contribution scores
        │
        └── { "risk": 84.37, "level": "Very High Risk", "shap_values": { ... } }
                │
                ▼
        index.html stores result in sessionStorage
                │
                ▼
        Browser navigates to results.html
                │
                ▼
        results.html renders animated score + SHAP breakdown + input summary
```

The model is loaded **once at server startup** and kept in memory, so every prediction request
is fast with no disk I/O on each call.

---

## ✨ Features

- **Two-page flow** — `index.html` handles input collection; `results.html` handles result display.
  Data is passed between pages via `sessionStorage`.
- **SHAP explainability** — every prediction includes per-feature SHAP contribution scores showing
  exactly which health indicators pushed the risk score up or down, making the model transparent
  and interpretable.
- **Rich input controls** — toggle switches for binary fields, segmented controls for general health
  and sex, a 13-button age-range grid, and numeric inputs for BMI and health day counts.
- **Dual-layer input validation** — the frontend prevents submission with missing or out-of-range
  values; the backend independently re-validates everything before the model is ever called.
- **All errors reported at once** — every validation violation is collected and returned in a single
  response, not one at a time.
- **Smart model caching** — on first run the model trains from the CSV and saves a `.pkl` file.
  Every subsequent server start loads from cache in under a second.
- **Class imbalance handling** — `scale_pos_weight` compensates for the CDC dataset's skewed class
  distribution (significantly more non-diabetic than diabetic samples).
- **No build step** — `index.html` and `results.html` are fully self-contained and open directly
  in any browser without a web server.
- **Custom typography** — uses Google Fonts (Instrument Serif + Geist) with a clean card-based layout.
- **Fully responsive** — the form layout adapts to mobile screen sizes via CSS grid breakpoints.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | CDC Behavioral Risk Factor Surveillance System (BRFSS) |
| **File** | `Diabetes Health Indicators Dataset export 2026-02-27 17-12-07.csv` |
| **Target column** | `Diabetes_012` (0 = No diabetes, 1 = Pre-diabetes, 2 = Diabetes) |
| **Preprocessing** | Pre-diabetes (1) and Diabetes (2) merged into a single positive class → binary classification |
| **Missing values** | Filled with column means (simple imputation) |
| **Train / Test split** | 80% / 20%, stratified by target class |

---

## 🖥️ Frontend Pages

### `index.html` — Health Questionnaire

The main input form, organized into four cards:

- **Vitals & Physical Health** — BMI (numeric input), General Health (5-option segmented control:
  Excellent → Poor), High Blood Pressure, Difficulty Walking, Physical Activity (toggle switches),
  and Mental / Physical poor health day counts (numeric inputs, 0–30).
- **Medical History** — Stroke history and Heart Disease / Heart Attack history (toggle switches).
- **Lifestyle** — Smoker status, Heavy Alcohol Use, Fruit intake, Vegetable intake (toggle switches).
- **Demographics** — Biological Sex (segmented control: Female / Male) and Age Group
  (13-button grid: 18–24 through 80+).

On submit, the form POSTs all 15 features to `http://localhost:5000/predict`, stores the full
JSON response (including SHAP values) in `sessionStorage`, and navigates to `results.html`.

### `results.html` — Prediction Result

Reads the prediction result from `sessionStorage` and displays:

- Animated risk percentage counter
- Color-coded circular progress meter
- Risk level badge (Low / Moderate / High / Very High)
- SHAP feature contribution breakdown — ranked list of which health indicators most influenced the score
- Full summary of all 15 submitted health indicator values
- "Retake Assessment" button to return to `index.html`

---

## 📋 Input Features & Validation Rules

| Feature | Type | Valid Range / Values | Description |
|---|---|---|---|
| `HighBP` | Binary | 0 or 1 | High blood pressure diagnosis |
| `BMI` | Continuous | 10 – 100 | Body Mass Index |
| `Smoker` | Binary | 0 or 1 | Smoked 100+ cigarettes in lifetime |
| `Stroke` | Binary | 0 or 1 | Ever had a stroke |
| `HeartDiseaseorAttack` | Binary | 0 or 1 | Coronary heart disease or heart attack |
| `PhysActivity` | Binary | 0 or 1 | Physical activity in past 30 days |
| `Fruits` | Binary | 0 or 1 | Eats fruit 1+ times/day |
| `Veggies` | Binary | 0 or 1 | Eats vegetables 1+ times/day |
| `HvyAlcoholConsump` | Binary | 0 or 1 | Heavy alcohol consumption |
| `GenHlth` | Ordinal | 1 – 5 | General health (1 = Excellent, 5 = Poor) |
| `MentHlth` | Continuous | 0 – 30 | Poor mental health days in past month |
| `PhysHlth` | Continuous | 0 – 30 | Poor physical health days in past month |
| `DiffWalk` | Binary | 0 or 1 | Difficulty walking or climbing stairs |
| `Sex` | Binary | 0 or 1 | Biological sex (0 = Female, 1 = Male) |
| `Age` | Ordinal | 1 – 13 | Age category (1 = 18–24, 13 = 80+) |

---

## 🎯 Risk Levels

| Risk Score | Label |
|---|---|
| 0% – 19% | 🟢 Low Risk |
| 20% – 49% | 🟡 Moderate Risk |
| 50% – 74% | 🟠 High Risk |
| 75% – 100% | 🔴 Very High Risk |

---

## 🔌 API Reference

### `POST /predict`

Runs the model, computes SHAP values, and returns a risk score with explainability data.

**Request body** (JSON):
```json
{
  "HighBP": 1, "BMI": 27.5, "Smoker": 0, "Stroke": 0,
  "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
  "Veggies": 1, "HvyAlcoholConsump": 0, "GenHlth": 2,
  "MentHlth": 3, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0, "Age": 4
}
```

**Success response** `200`:
```json
{
  "risk": 12.45,
  "level": "Low Risk",
  "shap_values": {
    "GenHlth": 0.312,
    "BMI": 0.198,
    "Age": 0.143,
    "HighBP": 0.091,
    "..."
  }
}
```

**Validation error** `400`:
```json
{
  "error": "Input validation failed (2 error(s)):\n  - 'BMI' must be between 10 and 100, got 150.\n  - 'GenHlth' must be between 1 and 5, got 9."
}
```

**Server error** `500`:
```json
{ "error": "Internal server error." }
```

---

### `GET /health`

Liveness check — confirms the server is running.

**Response** `200`:
```json
{ "status": "ok" }
```

---

### `GET /limits`

Returns the valid input range for every feature. The frontend calls this on load to enforce
input constraints dynamically without hard-coding them separately from the model definition.

**Response** `200` (excerpt):
```json
{
  "BMI":    { "type": "continuous", "min": 10,  "max": 100, "desc": "Body Mass Index" },
  "HighBP": { "type": "binary",     "allowed": [0, 1],      "desc": "High blood pressure flag (0 = No, 1 = Yes)" },
  "Age":    { "type": "ordinal",    "min": 1,   "max": 13,  "desc": "Age category (1 = 18-24, 13 = 80+)" }
}
```

---

## 🛠️ Setup & Installation

### Prerequisites

- **Python 3.10** or higher
- The CDC BRFSS dataset CSV placed in the **same directory** as `app.py`
- A modern browser (Chrome, Firefox, Edge, Safari)

### 1. Clone the repository

```bash
git clone https://github.com/medomario4-star/Diabetes-Prediction-AI.git
cd Diabetes-Prediction-AI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Full dependency list:

```
flask>=3.0
flask-cors>=4.0
xgboost>=2.0
scikit-learn>=1.4
shap>=0.44
pandas>=2.0
numpy>=1.26
joblib>=1.3
```

---

## 🚀 Running the App

### Step 1 — Start the Flask backend

```bash
python app.py
```

You will see output similar to:

```
Loading model…
Loading model from cache…        ← (or "training model…" on first run)
Model loaded from cache.
Model ready.
 * Running on http://127.0.0.1:5000
```

> **First run:** The model trains automatically (~30–60 seconds depending on your machine)
> and saves `diabetes_risk_model.pkl`. All subsequent starts load from cache in under a second.

### Step 2 — Open the frontend

Open `index.html` directly in your browser — **no web server needed**:

```
File → Open File → index.html
```

Or on Windows, simply **double-click** `index.html`.

### Step 3 — Use the app

1. Fill in all 15 health indicators across the four cards.
2. Click **Assess My Risk**.
3. The browser navigates to `results.html` and displays your animated risk score, risk level badge,
   SHAP feature contribution breakdown, and a full summary of your submitted inputs.

---

## 📈 Model Performance

Evaluated on a held-out **20% test split** (stratified):

| Metric | Score |
|---|---|
| ROC-AUC | **0.816** |
| F1 Score (weighted) | **0.749** |
| Approximate Accuracy | **~75%** |

**Confusion matrix** (on test set):

```
                  Predicted No    Predicted Yes
Actual No             29,876         12,865
Actual Yes             1,775          6,220
```

The model uses `scale_pos_weight` to handle **class imbalance** — the dataset contains significantly
more non-diabetic samples than diabetic ones. Without this correction the classifier would be biased
toward always predicting "no diabetes".

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost (`XGBClassifier`) |
| Explainability | SHAP (`TreeExplainer`) — per-prediction feature contributions |
| Preprocessing | scikit-learn (`StandardScaler`, `train_test_split`) |
| Backend | Python 3 + Flask + flask-cors |
| Frontend | Vanilla HTML / CSS / JavaScript — two pages, no framework |
| Fonts | Google Fonts — Instrument Serif + Geist |
| Result passing | Browser `sessionStorage` |
| Model persistence | joblib (`.pkl` cache file) |
| Dataset | CDC BRFSS (Behavioral Risk Factor Surveillance System) |

---

## ⚠️ Disclaimer

> This tool is intended **for educational purposes only** and does not constitute medical advice,
> diagnosis, or treatment.
> The underlying model has an approximate **accuracy of 75%**, meaning it may produce incorrect
> results in 1 out of every 4 cases.
> A high or low risk score does **not** confirm or rule out a diabetes diagnosis.
> Always consult a **qualified healthcare provider** for any medical concerns or before making
> health-related decisions.
