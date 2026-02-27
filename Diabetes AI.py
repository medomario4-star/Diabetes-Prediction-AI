# ==============================
# Diabetes Risk Prediction Model
# CDC BRFSS 2015 Dataset (Binary Risk)
# ==============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("Diabetes Health Indicators Dataset export 2026-02-27 17-12-07.csv")
print("Dataset shape:", df.shape)

# ==============================
# 2. FIX TARGET TO BINARY
# ==============================
# Convert 2 (diabetes) and 1 (prediabetes) into 1
df['Diabetes_012'] = df['Diabetes_012'].replace({2:1})

# ==============================
# 3. SELECT FEATURES
# ==============================
features = [
    'HighBP','HighChol','BMI','Smoker','Stroke','HeartDiseaseorAttack',
    'PhysActivity','Fruits','Veggies','HvyAlcoholConsump','GenHlth',
    'MentHlth','PhysHlth','DiffWalk','Sex','Age'
]
target = 'Diabetes_012'

X = df[features]
y = df[target]

# ==============================
# 4. TRAIN TEST SPLIT (stratified)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 5. HANDLE CLASS IMBALANCE
# ==============================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# ==============================
# 6. BUILD MODEL PIPELINE
# ==============================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, features)]
)

base_model = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    random_state=42, class_weight=class_weight_dict
)

model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', base_model)])

# ==============================
# 7. CALIBRATE MODEL
# ==============================
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)

# ==============================
# 8. TRAIN
# ==============================
calibrated_model.fit(X_train, y_train)

# EVALUATION
y_pred = calibrated_model.predict(X_test)
y_prob = calibrated_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# F1 Score
f1 = f1_score(y_test, y_pred, zero_division=0)
print("F1 Score:", f1)
# ==============================
# 10. SAVE MODEL
# ==============================
joblib.dump(calibrated_model, "diabetes_risk_model.pkl")
print("\nModel saved as diabetes_risk_model.pkl")

# ==============================
# 11. RISK SCORE FUNCTION
# ==============================
def predict_risk(input_data):
    """
    input_data: dictionary with same features as 'features' list
    Returns risk percentage (0-100) and risk level
    """
    input_df = pd.DataFrame([input_data])
    input_df = input_df[features]  # Ensure correct order
    prob = calibrated_model.predict_proba(input_df)[:, 1][0]
    risk_percentage = prob * 100

    if risk_percentage < 20:
        level = "Low Risk"
    elif risk_percentage < 50:
        level = "Moderate Risk"
    elif risk_percentage < 75:
        level = "High Risk"
    else:
        level = "Very High Risk"

    return round(risk_percentage, 2), level

# ==============================
# 12. EXAMPLE TEST
# ==============================
example_user = {
    'HighBP': 0,'HighChol': 1,'BMI': 32,'Smoker': 1,'Stroke': 0,
    'HeartDiseaseorAttack': 0,'PhysActivity': 0,'Fruits': 0,'Veggies': 0,
    'HvyAlcoholConsump': 0,'GenHlth': 4,'MentHlth': 10,'PhysHlth': 5,
    'DiffWalk': 1,'Sex': 1,'Age': 30
}

risk, level = predict_risk(example_user)
print("\nExample User Risk:")
print("Risk Score:", risk, "%")
print("Risk Level:", level)