"""
app.py — Flask backend for the Diabetes Risk Predictor
-------------------------------------------------------
This file exposes the trained XGBoost model as a REST API so the
frontend (index.html) can send user inputs and receive a risk score.

Endpoints:
    POST /predict   — receives 15 health-indicator values as JSON,
                      validates them, runs the model, and returns
                      { "risk": <float>, "level": <str> }

    GET  /health    — simple liveness check, returns { "status": "ok" }

    GET  /limits    — returns the valid input range for every feature
                      so the frontend can do its own boundary checks

Run:
    python app.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify   # Flask core: app factory, request parser, JSON responses
from flask_cors import CORS                 # Adds CORS headers so the browser doesn't block requests
import traceback                            # Used to print full stack traces on unexpected errors

# Import the model helpers and constants defined in diabetes_risk_model.py
from diabetes_risk_model import (
    load_model,       # Loads the model bundle from disk (or trains it if missing)
    predict_risk,     # Runs validation + inference and returns (risk_pct, risk_label)
    validate_inputs,  # Standalone validator — used here only for the type-casting step
    CACHE_FILE,       # Path to the saved model pickle file
    CSV_FILE,         # Path to the training dataset CSV
    FEATURE_LIMITS,   # Dict describing valid types and ranges for every feature
)

# ── App setup ─────────────────────────────────────────────────────────────────

# Create the Flask application instance
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) for all routes.
# This is required because the HTML frontend is opened directly in the browser
# as a file (or served from a different port), which counts as a different
# "origin" from the Flask server running on localhost:5000.
CORS(app)

# ── Model loading ─────────────────────────────────────────────────────────────

# Load the model once when the server starts — not on every request.
# load_model() checks for a cached .pkl file first; if it doesn't exist,
# it trains the model from scratch using the CSV and then saves the cache.
# Keeping the bundle in a module-level variable means it stays in memory
# and is shared across all incoming requests without reloading.
print("Loading model…")
bundle = load_model(CACHE_FILE, CSV_FILE)
print("Model ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """
    Liveness check endpoint.
    Useful for confirming the server is up before making prediction requests.
    Returns: 200 { "status": "ok" }
    """
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Expects a JSON body containing all 15 feature keys, e.g.:
        {
            "HighBP": 1, "BMI": 27.5, "Smoker": 0, "Stroke": 0,
            "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
            "Veggies": 1, "HvyAlcoholConsump": 0, "GenHlth": 2,
            "MentHlth": 3, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0, "Age": 4
        }

    Returns:
        200  { "risk": 84.37, "level": "Very High Risk" }  — on success
        400  { "error": "Input validation failed…" }       — on bad input
        500  { "error": "Internal server error." }         — on unexpected crash
    """

    # Parse the request body as JSON.
    # force=True  → treat the body as JSON even if Content-Type header is missing.
    # silent=True → return None instead of raising an exception on malformed JSON.
    data = request.get_json(force=True, silent=True)

    # If the body is empty or not valid JSON, reject early with a clear message.
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # ── Type coercion ─────────────────────────────────────────────────────────
    # JSON numbers arrive as Python int or float, but HTML form data and some
    # clients may send everything as strings. We normalise everything to float
    # first, then convert binary/ordinal features back to int where needed.

    typed = {}
    for key, value in data.items():
        try:
            typed[key] = float(value)   # covers "27.5", 27.5, 27, "1", True, etc.
        except (TypeError, ValueError):
            typed[key] = value          # leave as-is; validate_inputs will flag it

    # Binary features use set membership checks ({0, 1}), and ordinal features
    # use integer comparisons, so they must be int — not float.
    # Build a set of all feature keys that need integer representation.
    binary_ordinal = {
        k for k, v in FEATURE_LIMITS.items()
        if v["type"] in ("binary", "ordinal")
    }
    for k in binary_ordinal:
        if k in typed:
            try:
                typed[k] = int(typed[k])    # e.g. 1.0 → 1, so {0,1} check passes
            except (TypeError, ValueError):
                pass                        # leave broken values for validate_inputs

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        # predict_risk() internally calls validate_inputs() first.
        # If any value is missing or out of range, it raises ValueError.
        # On success it returns a (risk_percentage, risk_label) tuple.
        risk, level = predict_risk(typed, bundle)

        # IMPORTANT: XGBoost's predict_proba returns numpy float32 values.
        # Python's json module cannot serialise numpy types directly, so we
        # explicitly cast risk to a native Python float before returning.
        return jsonify({"risk": float(risk), "level": level})

    except ValueError as e:
        # Validation errors are user mistakes — return 400 Bad Request
        # and pass the full error message back so the frontend can display it.
        return jsonify({"error": str(e)}), 400

    except Exception:
        # Any other unexpected error (model bug, memory issue, etc.) — log the
        # full traceback to the console for debugging and return a generic 500.
        traceback.print_exc()
        return jsonify({"error": "Internal server error."}), 500


@app.route("/limits", methods=["GET"])
def limits():
    """
    Returns the valid input range for every feature as JSON.

    The frontend can call this endpoint on page load to dynamically
    build input constraints (min/max attributes, allowed values) without
    hard-coding them separately from the model definition.

    Returns a dict like:
        {
            "BMI":    { "type": "continuous", "min": 10, "max": 100, "desc": "…" },
            "HighBP": { "type": "binary", "allowed": [0, 1], "desc": "…" },
            "Age":    { "type": "ordinal", "min": 1, "max": 13, "desc": "…" },
            ...
        }
    """
    serialisable = {}

    for feature, meta in FEATURE_LIMITS.items():
        # Start with the fields that are common to all feature types
        entry = {
            "type": meta["type"],
            "desc": meta["desc"],
        }

        if meta["type"] == "binary":
            # Convert the Python set {0, 1} to a sorted list — sets are not
            # JSON serialisable, but lists are.
            entry["allowed"] = sorted(meta["allowed"])
        else:
            # For continuous and ordinal features, expose the numeric bounds
            entry["min"] = meta["min"]
            entry["max"] = meta["max"]

        serialisable[feature] = entry

    return jsonify(serialisable)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # debug=True enables:
    #   - Auto-reload when source files change (no need to restart manually)
    #   - The interactive Werkzeug debugger in the browser on unhandled errors
    # Never use debug=True in production — it exposes internal code to users.
    app.run(debug=True, port=5000)