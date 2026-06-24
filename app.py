"""
app.py — Flask backend for the Diabetes Risk Predictor
-------------------------------------------------------
Endpoints:
    POST /predict   — validates inputs, runs the model, returns risk score
                      + top 3 SHAP feature contributions
    GET  /health    — liveness check
    GET  /limits    — valid input ranges for all 15 features

Run:  python app.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import numpy as np
import pandas as pd
import shap
from flasgger import Swagger
from diabetes_risk_model import (
    load_model,
    predict_risk,
    validate_inputs,
    CACHE_FILE,
    CSV_FILE,
    FEATURE_LIMITS,
    FEATURES,
)

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)
Swagger(app)

# ── Model loading ─────────────────────────────────────────────────────────────

# Load model once at startup and build the SHAP explainer once.
# TreeExplainer is the correct explainer for XGBoost — it uses the exact
# Shapley values from the tree structure rather than approximations.
print("Loading model…")
bundle = load_model(CACHE_FILE, CSV_FILE)
print("Building SHAP explainer…")
explainer = shap.TreeExplainer(bundle["model"])
print("Ready.")

# ── Human-readable feature labels ─────────────────────────────────────────────

FEATURE_LABELS = {
    "HighBP":               "High Blood Pressure",
    "BMI":                  "Body Mass Index (BMI)",
    "Smoker":               "Smoking History",
    "Stroke":               "Stroke History",
    "HeartDiseaseorAttack": "Heart Disease / Attack",
    "PhysActivity":         "Physical Activity",
    "Fruits":               "Daily Fruit Intake",
    "Veggies":              "Daily Vegetable Intake",
    "HvyAlcoholConsump":    "Heavy Alcohol Use",
    "GenHlth":              "General Health Rating",
    "MentHlth":             "Poor Mental Health Days",
    "PhysHlth":             "Poor Physical Health Days",
    "DiffWalk":             "Difficulty Walking",
    "Sex":                  "Biological Sex",
    "Age":                  "Age Group",
}

# ── SHAP helper ───────────────────────────────────────────────────────────────

def get_top_factors(typed_input: dict, n: int = 3) -> list[dict]:
    """
    Compute SHAP values for one prediction and return the top-n features
    that pushed the risk score upward (positive SHAP = increases risk).

    Each entry in the returned list:
        {
            "feature":     "BMI",
            "label":       "Body Mass Index (BMI)",
            "value":       38.0,          # the user's actual input
            "shap":        0.42,          # raw SHAP value (log-odds space)
            "contribution": 18.3          # percentage-share of total positive SHAP
        }

    We only report features with a positive SHAP value because those are
    the ones driving the risk UP — negative SHAP values are protective factors.
    """
    model   = bundle["model"]
    scaler  = bundle["scaler"]

    # Build a single-row DataFrame in the correct column order
    input_df     = pd.DataFrame([typed_input])[FEATURES]
    input_scaled = scaler.transform(input_df)

    # Compute SHAP values — shape: (1, n_features)
    # [1] selects values for the positive class (diabetes = 1)
    shap_values = explainer.shap_values(input_scaled)

    # XGBoost TreeExplainer returns an array directly (not a list of two)
    # but handle both cases defensively
    if isinstance(shap_values, list):
        sv = shap_values[1][0]   # positive class, first (only) row
    else:
        sv = shap_values[0]      # single array, first row

    # Pair each feature with its SHAP value
    pairs = list(zip(FEATURES, sv))

    # Keep only risk-increasing factors (positive SHAP)
    positive = [(feat, float(val)) for feat, val in pairs if val > 0]

    if not positive:
        # Edge case: all features are protective — return top by abs value
        positive = [(feat, abs(float(val))) for feat, val in pairs]

    # Sort descending and take top-n
    positive.sort(key=lambda x: x[1], reverse=True)
    top = positive[:n]

    # Express each factor's SHAP as a % share of the total positive SHAP
    total_positive = sum(v for _, v in positive)

    result = []
    for feat, sv_val in top:
        contribution = round((sv_val / total_positive) * 100, 1) if total_positive else 0
        result.append({
            "feature":      feat,
            "label":        FEATURE_LABELS.get(feat, feat),
            "value":        float(typed_input.get(feat, 0)),
            "shap":         round(sv_val, 4),
            "contribution": contribution,
        })

    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """
        Health check
        ---
        responses:
          200:
            description: Server is running
        """
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict diabetes risk
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          properties:
            HighBP:
              type: integer
              example: 1
            BMI:
              type: number
              example: 28.5
            Age:
              type: integer
              example: 5
    responses:
      200:
        description: Risk score returned successfully
      400:
        description: Invalid input
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # ── Type coercion ──────────────────────────────────────────────────────────
    typed = {}
    for key, value in data.items():
        try:
            typed[key] = float(value)
        except (TypeError, ValueError):
            typed[key] = value

    binary_ordinal = {
        k for k, v in FEATURE_LIMITS.items()
        if v["type"] in ("binary", "ordinal")
    }
    for k in binary_ordinal:
        if k in typed:
            try:
                typed[k] = int(typed[k])
            except (TypeError, ValueError):
                pass

    # ── Inference + SHAP ──────────────────────────────────────────────────────
    try:
        risk, level = predict_risk(typed, bundle)
        factors     = get_top_factors(typed, n=3)

        return jsonify({
            "risk":    float(risk),
            "level":   level,
            "factors": factors,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error."}), 500


@app.route("/limits", methods=["GET"])
def limits():
    """Returns the valid input range for every feature."""
    serialisable = {}
    for feature, meta in FEATURE_LIMITS.items():
        entry = {"type": meta["type"], "desc": meta["desc"]}
        if meta["type"] == "binary":
            entry["allowed"] = sorted(meta["allowed"])
        else:
            entry["min"] = meta["min"]
            entry["max"] = meta["max"]
        serialisable[feature] = entry
    return jsonify(serialisable)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
