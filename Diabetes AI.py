import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Constants ────────────────────────────────────────────────────────────────

CACHE_FILE = "diabetes_risk_model.pkl"
CSV_FILE   = "Diabetes Health Indicators Dataset export 2026-02-27 17-12-07.csv"
TEST_SIZE  = 0.2
RANDOM_STATE = 42

# Features selected from the dataset to train on.
# These are health indicators collected via survey (CDC BRFSS dataset).
FEATURES = [
    "HighBP",               # Has high blood pressure? (0/1)
    "BMI",                  # Body Mass Index (continuous)
    "Smoker",               # Has smoked 100+ cigarettes in lifetime? (0/1)
    "Stroke",               # Ever had a stroke? (0/1)
    "HeartDiseaseorAttack", # Has/had coronary heart disease or MI? (0/1)
    "PhysActivity",         # Physical activity in past 30 days? (0/1)
    "Fruits",               # Consumes fruit 1+ times/day? (0/1)
    "Veggies",              # Consumes vegetables 1+ times/day? (0/1)
    "HvyAlcoholConsump",    # Heavy alcohol consumption? (0/1)
    "GenHlth",              # General health rating (1=Excellent … 5=Poor)
    "MentHlth",             # Poor mental health days in past month (0–30)
    "PhysHlth",             # Poor physical health days in past month (0–30)
    "DiffWalk",             # Difficulty walking or climbing stairs? (0/1)
    "Sex",                  # Biological sex (0=Female, 1=Male)
    "Age",                  # Age category (1=18-24, 2=25-29, … 13=80+)
]

TARGET = "Diabetes_012"

# Risk thresholds for human-readable labels
RISK_THRESHOLDS = [
    (20,  "Low Risk"),
    (50,  "Moderate Risk"),
    (75,  "High Risk"),
    (101, "Very High Risk"),  # 101 acts as a catch-all upper bound
]

# ── Data loading & preprocessing ─────────────────────────────────────────────

def load_and_preprocess(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the CSV, fill missing values, and binarise the target column.

    The original target has three classes (0=no diabetes, 1=pre-diabetes,
    2=diabetes). We collapse pre-diabetes and diabetes into a single
    positive class so this becomes a binary classification problem.
    """
    df = pd.read_csv(csv_path)

    # Replace missing values with column means (simple imputation)
    df.fillna(df.mean(), inplace=True)

    # Merge class 2 (diabetes) into class 1 (pre-diabetes) → binary target
    df[TARGET] = df[TARGET].replace({2: 1})

    return df[FEATURES], df[TARGET]


# ── Model training ────────────────────────────────────────────────────────────

def build_model() -> XGBClassifier:
    """Return a configured (but untrained) XGBoost classifier."""
    return XGBClassifier(
        n_estimators=100,   # Number of boosting rounds
        learning_rate=0.1,  # Step size shrinkage to prevent overfitting
        max_depth=5,        # Maximum depth of each tree
        random_state=RANDOM_STATE,
        eval_metric="logloss",  # Log-loss for binary classification
    )


def evaluate(model: XGBClassifier, X_test_scaled, y_test) -> None:
    """Print key evaluation metrics for the trained model."""
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class

    print("ROC-AUC Score  :", roc_auc_score(y_test, y_prob))
    print("F1 Score       :", f1_score(y_test, y_pred, average="weighted"))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def train_and_save_model(csv_path: str, cache_path: str) -> dict:
    """Train the model from scratch and persist it to disk.

    Returns the model bundle (model, scaler, feature list) so the
    caller doesn't have to re-load from disk immediately.
    """
    X, y = load_and_preprocess(csv_path)

    # Stratified split preserves the class ratio in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Standardise features to zero mean, unit variance.
    # Fit ONLY on training data to avoid data leakage into the test set.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = build_model()
    model.fit(X_train_scaled, y_train)

    evaluate(model, X_test_scaled, y_test)

    # Bundle everything the predictor needs at inference time
    model_bundle = {"model": model, "scaler": scaler, "features": FEATURES}
    joblib.dump(model_bundle, cache_path)
    print(f"\nModel saved to {cache_path}")

    return model_bundle


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(cache_path: str, csv_path: str) -> dict:
    """Load the model bundle from cache, training first if no cache exists."""
    try:
        print("Loading model from cache…")
        bundle = joblib.load(cache_path)
        print("Model loaded from cache.")
    except FileNotFoundError:
        print("No cache found — training model…")
        bundle = train_and_save_model(csv_path, cache_path)
        print("Model loaded after training.")

    return bundle


# ── Inference ─────────────────────────────────────────────────────────────────

def get_risk_label(risk_pct: float) -> str:
    """Map a risk percentage to a human-readable label."""
    for threshold, label in RISK_THRESHOLDS:
        if risk_pct < threshold:
            return label


def predict_risk(input_data: dict, bundle: dict) -> tuple[float, str]:
    """Predict diabetes risk from a dictionary of health indicators.

    Args:
        input_data: Feature values keyed by feature name.
        bundle:     Model bundle returned by load_model().

    Returns:
        (risk_percentage, risk_label) — e.g. (34.7, "Moderate Risk")
    """
    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]

    # Build a single-row DataFrame in the exact column order the model expects
    input_df = pd.DataFrame([input_data])[features]
    input_scaled = scaler.transform(input_df)

    # predict_proba returns [[p_negative, p_positive]]; we want p_positive
    prob = model.predict_proba(input_scaled)[0, 1]
    risk_pct = round(prob * 100, 2)

    return risk_pct, get_risk_label(risk_pct)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load (or train) the model
    bundle = load_model(CACHE_FILE, CSV_FILE)

    # Example: healthy female, aged 25–29
    example_user = {
        'HighBP': 1, 'BMI': 27, 'Smoker': 0, 'Stroke': 0,
        'HeartDiseaseorAttack': 0, 'PhysActivity': 1, 'Fruits': 0, 'Veggies': 1,
        'HvyAlcoholConsump': 0, 'GenHlth': 3, 'MentHlth': 2, 'PhysHlth': 2,
        'DiffWalk': 0, 'Sex': 1, 'Age': 6
    }

    risk, level = predict_risk(example_user, bundle)
    print("\nExample User Risk:")
    print("Risk Score:", risk, "%")
    print("Risk Level:", level)
