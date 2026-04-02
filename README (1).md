# Diabetes Risk Predictor

A machine learning web application that estimates a user's diabetes risk from 15 health indicators. Built with an XGBoost classifier trained on the CDC BRFSS survey dataset, a Flask REST API backend, and a standalone HTML/CSS/JS frontend — no framework required.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Features](#features)
- [Input Features & Validation Rules](#input-features--validation-rules)
- [Risk Levels](#risk-levels)
- [API Reference](#api-reference)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer)

---

## Overview

The user fills in a health questionnaire in the browser. The frontend sends the answers to the Flask backend, which validates the inputs, runs them through the trained model, and returns a risk percentage (0–100%) along with a human-readable risk level. The result is displayed with an animated score counter and a color-coded progress meter.

---

## Project Structure

```
your-project/
│
├── diabetes_risk_model.py        # ML core: data loading, training, validation, inference
├── app.py                        # Flask REST API (3 endpoints)
├── index.html                    # Frontend — single HTML file, no build step needed
├── requirements.txt              # Python dependencies
│
├── diabetes_risk_model.pkl       # Auto-generated on first run — cached trained model
└── Diabetes Health Indicators Dataset export 2026-02-27 17-12-07.csv   # Training data
```

> The `.pkl` cache file is created automatically the first time you run the app.
> You do **not** need to train the model manually.

---

## How It Works

```
Browser (index.html)
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
        │       └── XGBClassifier.predict_proba()
        │
        └── { "risk": 84.37, "level": "Very High Risk" }
                │
                ▼
        Browser renders animated result
```

The model is loaded **once at server startup** and kept in memory, so every prediction request is fast with no disk I/O.

---

## Features

- **Input validation on both sides** — the frontend checks for completeness before submitting; the backend re-validates all values against defined type and range rules before the model is ever called.
- **All errors reported at once** — if multiple fields are invalid, every violation is returned in a single response rather than one at a time.
- **Smart model caching** — on first run the model trains from the CSV and saves a `.pkl` file. Every subsequent run loads from cache in under a second.
- **Class imbalance handling** — `scale_pos_weight` compensates for the dataset's imbalanced class distribution (more non-diabetic than diabetic samples).
- **No build step** — `index.html` is fully self-contained and opens directly in any browser.

---

## Input Features & Validation Rules

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

## Risk Levels

| Risk Score | Label |
|---|---|
| 0% – 19% | 🟢 Low Risk |
| 20% – 49% | 🟡 Moderate Risk |
| 50% – 74% | 🟠 High Risk |
| 75% – 100% | 🔴 Very High Risk |

---

## API Reference

### `POST /predict`

Runs the model and returns a risk score.

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
{ "risk": 12.45, "level": "Low Risk" }
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

Returns the valid range for every feature. The frontend uses this to enforce input constraints dynamically.

**Response** `200` (excerpt):
```json
{
  "BMI":    { "type": "continuous", "min": 10,  "max": 100, "desc": "Body Mass Index" },
  "HighBP": { "type": "binary",     "allowed": [0, 1],       "desc": "High blood pressure flag (0 = No, 1 = Yes)" },
  "Age":    { "type": "ordinal",    "min": 1,   "max": 13,   "desc": "Age category (1 = 18-24, 13 = 80+)" }
}
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- The CDC BRFSS dataset CSV file placed in the same directory as `app.py`

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd your-project
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

**`requirements.txt` contents:**
```
flask>=3.0
flask-cors>=4.0
xgboost>=2.0
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
joblib>=1.3
```

---

## Running the App

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

> On first run the model trains automatically (~30–60 seconds depending on your machine)
> and saves `diabetes_risk_model.pkl`. All subsequent starts load from cache instantly.

### Step 2 — Open the frontend

Open `index.html` directly in your browser — no web server needed:

```
File → Open File → index.html
```

Or on Windows, just double-click `index.html`.

### Step 3 — Use the app

Fill in all 15 health indicators and click **Calculate Risk**. The result appears below the form with an animated score, risk level badge, and progress meter.

---

## Model Performance

Evaluated on a held-out 20% test split (stratified):

| Metric | Score |
|---|---|
| ROC-AUC | 0.816 |
| F1 Score (weighted) | 0.749 |

**Confusion matrix** (on test set):

```
                Predicted No    Predicted Yes
Actual No           29,876         12,865
Actual Yes           1,775          6,220
```

The model uses `scale_pos_weight` to handle class imbalance — the dataset contains significantly more non-diabetic samples than diabetic ones, and without this correction the classifier would be biased toward predicting "no diabetes" for everyone.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost (`XGBClassifier`) |
| Preprocessing | scikit-learn (`StandardScaler`, `train_test_split`) |
| Backend | Python 3 + Flask + flask-cors |
| Frontend | Vanilla HTML / CSS / JavaScript (no framework) |
| Model persistence | joblib |
| Dataset | CDC BRFSS (Behavioral Risk Factor Surveillance System) |

---

## Disclaimer

> This tool is for **informational and educational purposes only**. It is not a
> substitute for professional medical advice, diagnosis, or treatment. The risk
> score is a statistical estimate based on population-level survey data and
> does not constitute a clinical assessment. Always consult a qualified
> healthcare provider regarding any health concerns.
