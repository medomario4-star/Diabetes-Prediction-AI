"""
tests.py — Automated test suite for the Diabetes Risk Predictor
================================================================
Covers three layers:
  1. Validation logic  (diabetes_risk_model.validate_inputs)
  2. Inference logic   (diabetes_risk_model.predict_risk, get_risk_label)
  3. Flask API         (all three endpoints via Flask test client)

Frontend / browser tests live in:
  → test_frontend.py  (Selenium, requires Chrome + Flask server running)

Run backend + API tests only (no browser needed):
    python -m pytest tests.py -v

Run a specific group:
    python -m pytest tests.py -v -k "validation"
    python -m pytest tests.py -v -k "api"
    python -m pytest tests.py -v -k "inference"

Run everything including frontend:
    python app.py          ← terminal 1 (keep running)
    python -m pytest tests.py test_frontend.py -v   ← terminal 2

Requirements:
    pip install pytest selenium webdriver-manager
    (flask, xgboost, scikit-learn, pandas already installed)
"""

import json
import pytest

# ── Shared test data ──────────────────────────────────────────────────────────

# A complete, valid set of inputs that should always pass validation
VALID_INPUT = {
    "HighBP":               1,
    "BMI":                  27.5,
    "Smoker":               0,
    "Stroke":               0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity":         1,
    "Fruits":               1,
    "Veggies":              1,
    "HvyAlcoholConsump":    0,
    "GenHlth":              2,
    "MentHlth":             3,
    "PhysHlth":             0,
    "DiffWalk":             0,
    "Sex":                  0,
    "Age":                  4,
}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Validation logic
# ══════════════════════════════════════════════════════════════════════════════

class TestValidation:
    """Tests for validate_inputs() in diabetes_risk_model.py."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from diabetes_risk_model import validate_inputs
        self.validate = validate_inputs

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_valid_input_passes(self):
        """A complete valid payload should raise no errors."""
        self.validate(VALID_INPUT)  # must not raise

    def test_bmi_boundary_min(self):
        """BMI exactly at its lower bound (10) should be accepted."""
        data = {**VALID_INPUT, "BMI": 10}
        self.validate(data)

    def test_bmi_boundary_max(self):
        """BMI exactly at its upper bound (100) should be accepted."""
        data = {**VALID_INPUT, "BMI": 100}
        self.validate(data)

    def test_age_boundary_min(self):
        """Age = 1 (lowest category) should be accepted."""
        data = {**VALID_INPUT, "Age": 1}
        self.validate(data)

    def test_age_boundary_max(self):
        """Age = 13 (80+) should be accepted."""
        data = {**VALID_INPUT, "Age": 13}
        self.validate(data)

    def test_genhlth_boundary_min(self):
        """GenHlth = 1 (Excellent) should be accepted."""
        data = {**VALID_INPUT, "GenHlth": 1}
        self.validate(data)

    def test_genhlth_boundary_max(self):
        """GenHlth = 5 (Poor) should be accepted."""
        data = {**VALID_INPUT, "GenHlth": 5}
        self.validate(data)

    def test_menthlth_zero(self):
        """MentHlth = 0 (no bad mental health days) should be accepted."""
        data = {**VALID_INPUT, "MentHlth": 0}
        self.validate(data)

    def test_menthlth_thirty(self):
        """MentHlth = 30 (all days) should be accepted."""
        data = {**VALID_INPUT, "MentHlth": 30}
        self.validate(data)

    def test_binary_zero_accepted(self):
        """All binary fields set to 0 should be accepted."""
        data = {**VALID_INPUT,
                "HighBP": 0, "Smoker": 0, "Stroke": 0,
                "HeartDiseaseorAttack": 0, "PhysActivity": 0,
                "Fruits": 0, "Veggies": 0, "HvyAlcoholConsump": 0,
                "DiffWalk": 0, "Sex": 0}
        self.validate(data)

    def test_binary_one_accepted(self):
        """All binary fields set to 1 should be accepted."""
        data = {**VALID_INPUT,
                "HighBP": 1, "Smoker": 1, "Stroke": 1,
                "HeartDiseaseorAttack": 1, "PhysActivity": 1,
                "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 1,
                "DiffWalk": 1, "Sex": 1}
        self.validate(data)

    # ── Binary violations ─────────────────────────────────────────────────────

    def test_highbp_invalid_value(self):
        """HighBP = 2 must raise ValueError."""
        with pytest.raises(ValueError, match="HighBP"):
            self.validate({**VALID_INPUT, "HighBP": 2})

    def test_smoker_negative(self):
        """Smoker = -1 must raise ValueError."""
        with pytest.raises(ValueError, match="Smoker"):
            self.validate({**VALID_INPUT, "Smoker": -1})

    def test_sex_invalid(self):
        """Sex = 3 must raise ValueError."""
        with pytest.raises(ValueError, match="Sex"):
            self.validate({**VALID_INPUT, "Sex": 3})

    def test_stroke_float_invalid(self):
        """Stroke = 0.5 is not in {0, 1}, must raise ValueError."""
        with pytest.raises(ValueError, match="Stroke"):
            self.validate({**VALID_INPUT, "Stroke": 0.5})

    # ── Continuous / ordinal violations ───────────────────────────────────────

    def test_bmi_below_min(self):
        """BMI = 9 (below 10) must raise ValueError."""
        with pytest.raises(ValueError, match="BMI"):
            self.validate({**VALID_INPUT, "BMI": 9})

    def test_bmi_above_max(self):
        """BMI = 101 (above 100) must raise ValueError."""
        with pytest.raises(ValueError, match="BMI"):
            self.validate({**VALID_INPUT, "BMI": 101})

    def test_age_zero(self):
        """Age = 0 (below minimum of 1) must raise ValueError."""
        with pytest.raises(ValueError, match="Age"):
            self.validate({**VALID_INPUT, "Age": 0})

    def test_age_above_max(self):
        """Age = 14 (above maximum of 13) must raise ValueError."""
        with pytest.raises(ValueError, match="Age"):
            self.validate({**VALID_INPUT, "Age": 14})

    def test_genhlth_above_max(self):
        """GenHlth = 6 must raise ValueError."""
        with pytest.raises(ValueError, match="GenHlth"):
            self.validate({**VALID_INPUT, "GenHlth": 6})

    def test_menthlth_above_max(self):
        """MentHlth = 31 must raise ValueError."""
        with pytest.raises(ValueError, match="MentHlth"):
            self.validate({**VALID_INPUT, "MentHlth": 31})

    def test_physhlth_negative(self):
        """PhysHlth = -1 must raise ValueError."""
        with pytest.raises(ValueError, match="PhysHlth"):
            self.validate({**VALID_INPUT, "PhysHlth": -1})

    # ── Missing features ──────────────────────────────────────────────────────

    def test_missing_single_feature(self):
        """Omitting BMI must raise ValueError mentioning 'BMI'."""
        data = {k: v for k, v in VALID_INPUT.items() if k != "BMI"}
        with pytest.raises(ValueError, match="BMI"):
            self.validate(data)

    def test_missing_multiple_features(self):
        """Omitting several features must raise ValueError and report all of them."""
        data = {k: v for k, v in VALID_INPUT.items()
                if k not in ("BMI", "Age", "GenHlth")}
        with pytest.raises(ValueError) as exc_info:
            self.validate(data)
        msg = str(exc_info.value)
        assert "BMI"     in msg
        assert "Age"     in msg
        assert "GenHlth" in msg

    def test_empty_input_reports_all_features(self):
        """Passing an empty dict must report all 15 missing features."""
        with pytest.raises(ValueError) as exc_info:
            self.validate({})
        msg = str(exc_info.value)
        assert "15" in msg  # error count in the message

    # ── Type violations ───────────────────────────────────────────────────────

    def test_string_value_rejected(self):
        """Passing a non-numeric string for BMI must raise ValueError."""
        with pytest.raises(ValueError, match="BMI"):
            self.validate({**VALID_INPUT, "BMI": "heavy"})

    def test_none_value_rejected(self):
        """Passing None for a feature must raise ValueError."""
        with pytest.raises(ValueError, match="HighBP"):
            self.validate({**VALID_INPUT, "HighBP": None})

    # ── Multiple errors at once ───────────────────────────────────────────────

    def test_multiple_errors_reported_together(self):
        """All violations in one payload must be reported in a single raise."""
        data = {**VALID_INPUT,
                "HighBP": 5,   # binary violation
                "BMI":    150, # continuous violation
                "Age":    0,   # ordinal violation
                }
        with pytest.raises(ValueError) as exc_info:
            self.validate(data)
        msg = str(exc_info.value)
        assert "HighBP" in msg
        assert "BMI"    in msg
        assert "Age"    in msg


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Inference logic
# ══════════════════════════════════════════════════════════════════════════════

class TestInference:
    """Tests for predict_risk() and get_risk_label() in diabetes_risk_model.py."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from diabetes_risk_model import predict_risk, get_risk_label, load_model, CACHE_FILE, CSV_FILE
        self.predict    = predict_risk
        self.risk_label = get_risk_label
        self.bundle     = load_model(CACHE_FILE, CSV_FILE)

    # ── get_risk_label ────────────────────────────────────────────────────────

    def test_label_low(self):
        assert self.risk_label(0)  == "Low Risk"
        assert self.risk_label(10) == "Low Risk"
        assert self.risk_label(19.99) == "Low Risk"

    def test_label_moderate(self):
        assert self.risk_label(20) == "Moderate Risk"
        assert self.risk_label(35) == "Moderate Risk"
        assert self.risk_label(49.99) == "Moderate Risk"

    def test_label_high(self):
        assert self.risk_label(50) == "High Risk"
        assert self.risk_label(62) == "High Risk"
        assert self.risk_label(74.99) == "High Risk"

    def test_label_very_high(self):
        assert self.risk_label(75)  == "Very High Risk"
        assert self.risk_label(90)  == "Very High Risk"
        assert self.risk_label(100) == "Very High Risk"

    # ── predict_risk output types ─────────────────────────────────────────────

    def test_returns_float_and_string(self):
        """predict_risk must return (float, str)."""
        risk, level = self.predict(VALID_INPUT, self.bundle)
        assert isinstance(risk,  float),  f"risk should be float, got {type(risk)}"
        assert isinstance(level, str),    f"level should be str, got {type(level)}"

    def test_risk_in_valid_range(self):
        """Risk score must always be between 0 and 100."""
        risk, _ = self.predict(VALID_INPUT, self.bundle)
        assert 0 <= risk <= 100, f"Risk {risk} is outside [0, 100]"

    def test_level_is_known_label(self):
        """Level must be one of the four defined risk labels."""
        KNOWN = {"Low Risk", "Moderate Risk", "High Risk", "Very High Risk"}
        _, level = self.predict(VALID_INPUT, self.bundle)
        assert level in KNOWN, f"Unknown level: {level!r}"

    def test_level_matches_score(self):
        """The returned label must correspond to the returned risk score."""
        risk, level = self.predict(VALID_INPUT, self.bundle)
        if risk < 20:
            assert level == "Low Risk"
        elif risk < 50:
            assert level == "Moderate Risk"
        elif risk < 75:
            assert level == "High Risk"
        else:
            assert level == "Very High Risk"

    def test_deterministic(self):
        """Same input must always produce the same output."""
        r1, l1 = self.predict(VALID_INPUT, self.bundle)
        r2, l2 = self.predict(VALID_INPUT, self.bundle)
        assert r1 == r2
        assert l1 == l2

    # ── High-risk profile ─────────────────────────────────────────────────────

    def test_high_risk_profile(self):
        """A clinically high-risk profile should score above 50%."""
        high_risk = {
            "HighBP": 1, "BMI": 40, "Smoker": 1, "Stroke": 1,
            "HeartDiseaseorAttack": 1, "PhysActivity": 0,
            "Fruits": 0, "Veggies": 0, "HvyAlcoholConsump": 0,
            "GenHlth": 5, "MentHlth": 20, "PhysHlth": 20,
            "DiffWalk": 1, "Sex": 1, "Age": 11,
        }
        risk, _ = self.predict(high_risk, self.bundle)
        assert risk > 50, f"Expected high risk > 50%, got {risk}%"

    # ── Validation blocks inference ───────────────────────────────────────────

    def test_invalid_input_raises_before_model(self):
        """predict_risk must raise ValueError on bad input, never touch the model."""
        bad = {**VALID_INPUT, "BMI": 999}
        with pytest.raises(ValueError, match="BMI"):
            self.predict(bad, self.bundle)

    def test_missing_feature_raises(self):
        """predict_risk must raise ValueError when a feature is missing."""
        incomplete = {k: v for k, v in VALID_INPUT.items() if k != "Age"}
        with pytest.raises(ValueError, match="Age"):
            self.predict(incomplete, self.bundle)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Flask API endpoints
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    """
    Create a Flask test client that shares one model bundle for the
    entire module — avoids reloading/retraining on every test class.
    """
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_returns_ok_status(self, client):
        data = res = client.get("/health").get_json()
        assert data["status"] == "ok"

    def test_content_type_json(self, client):
        res = client.get("/health")
        assert "application/json" in res.content_type


class TestLimitsEndpoint:
    """Tests for GET /limits."""

    def test_returns_200(self, client):
        res = client.get("/limits")
        assert res.status_code == 200

    def test_all_15_features_present(self, client):
        data = client.get("/limits").get_json()
        from diabetes_risk_model import FEATURES
        for f in FEATURES:
            assert f in data, f"Feature '{f}' missing from /limits response"

    def test_binary_features_have_allowed(self, client):
        data = client.get("/limits").get_json()
        assert "allowed" in data["HighBP"]
        assert sorted(data["HighBP"]["allowed"]) == [0, 1]

    def test_continuous_features_have_min_max(self, client):
        data = client.get("/limits").get_json()
        bmi = data["BMI"]
        assert "min" in bmi and "max" in bmi
        assert bmi["min"] == 10
        assert bmi["max"] == 100

    def test_ordinal_features_have_min_max(self, client):
        data = client.get("/limits").get_json()
        age = data["Age"]
        assert age["min"] == 1
        assert age["max"] == 13

    def test_every_feature_has_desc(self, client):
        data = client.get("/limits").get_json()
        for feature, meta in data.items():
            assert "desc" in meta, f"'{feature}' is missing 'desc'"


class TestPredictEndpoint:
    """Tests for POST /predict."""

    # ── Success cases ─────────────────────────────────────────────────────────

    def test_valid_input_returns_200(self, client):
        res = client.post("/predict",
                          data=json.dumps(VALID_INPUT),
                          content_type="application/json")
        assert res.status_code == 200

    def test_response_has_risk_and_level(self, client):
        data = client.post("/predict",
                           data=json.dumps(VALID_INPUT),
                           content_type="application/json").get_json()
        assert "risk"  in data
        assert "level" in data

    def test_risk_is_float(self, client):
        data = client.post("/predict",
                           data=json.dumps(VALID_INPUT),
                           content_type="application/json").get_json()
        assert isinstance(data["risk"], float), \
            f"Expected float, got {type(data['risk'])}"

    def test_risk_in_range(self, client):
        data = client.post("/predict",
                           data=json.dumps(VALID_INPUT),
                           content_type="application/json").get_json()
        assert 0 <= data["risk"] <= 100

    def test_level_is_valid_label(self, client):
        data = client.post("/predict",
                           data=json.dumps(VALID_INPUT),
                           content_type="application/json").get_json()
        KNOWN = {"Low Risk", "Moderate Risk", "High Risk", "Very High Risk"}
        assert data["level"] in KNOWN

    def test_deterministic_across_requests(self, client):
        """Two identical POST requests must return the same risk score."""
        payload = json.dumps(VALID_INPUT)
        r1 = client.post("/predict", data=payload, content_type="application/json").get_json()
        r2 = client.post("/predict", data=payload, content_type="application/json").get_json()
        assert r1["risk"]  == r2["risk"]
        assert r1["level"] == r2["level"]

    def test_accepts_request_without_content_type(self, client):
        """force=True in get_json means Content-Type header is not required."""
        res = client.post("/predict", data=json.dumps(VALID_INPUT))
        assert res.status_code == 200

    def test_string_numbers_coerced(self, client):
        """String-encoded numbers (e.g. from HTML forms) should be accepted."""
        stringified = {k: str(v) for k, v in VALID_INPUT.items()}
        res = client.post("/predict",
                          data=json.dumps(stringified),
                          content_type="application/json")
        assert res.status_code == 200

    # ── Validation error cases (400) ──────────────────────────────────────────

    def test_empty_body_returns_400(self, client):
        res = client.post("/predict", data="", content_type="application/json")
        assert res.status_code == 400

    def test_invalid_json_returns_400(self, client):
        res = client.post("/predict", data="not-json", content_type="application/json")
        assert res.status_code == 400

    def test_missing_feature_returns_400(self, client):
        incomplete = {k: v for k, v in VALID_INPUT.items() if k != "BMI"}
        res = client.post("/predict",
                          data=json.dumps(incomplete),
                          content_type="application/json")
        assert res.status_code == 400

    def test_missing_feature_error_message(self, client):
        incomplete = {k: v for k, v in VALID_INPUT.items() if k != "BMI"}
        data = client.post("/predict",
                           data=json.dumps(incomplete),
                           content_type="application/json").get_json()
        assert "error" in data
        assert "BMI" in data["error"]

    def test_out_of_range_bmi_returns_400(self, client):
        bad = {**VALID_INPUT, "BMI": 999}
        res = client.post("/predict",
                          data=json.dumps(bad),
                          content_type="application/json")
        assert res.status_code == 400

    def test_invalid_binary_returns_400(self, client):
        bad = {**VALID_INPUT, "HighBP": 5}
        res = client.post("/predict",
                          data=json.dumps(bad),
                          content_type="application/json")
        assert res.status_code == 400

    def test_invalid_age_returns_400(self, client):
        bad = {**VALID_INPUT, "Age": 0}
        res = client.post("/predict",
                          data=json.dumps(bad),
                          content_type="application/json")
        assert res.status_code == 400

    def test_error_response_has_error_key(self, client):
        """All 400 responses must include an 'error' key, not 'risk'."""
        bad = {**VALID_INPUT, "BMI": -5}
        data = client.post("/predict",
                           data=json.dumps(bad),
                           content_type="application/json").get_json()
        assert "error" in data
        assert "risk"  not in data

    def test_multiple_invalid_fields_all_reported(self, client):
        """All violations must appear in the error message."""
        bad = {**VALID_INPUT, "HighBP": 9, "BMI": 200, "Age": 99}
        data = client.post("/predict",
                           data=json.dumps(bad),
                           content_type="application/json").get_json()
        assert "HighBP" in data["error"]
        assert "BMI"    in data["error"]
        assert "Age"    in data["error"]

    # ── Boundary values at API level ──────────────────────────────────────────

    def test_bmi_at_minimum_boundary(self, client):
        """BMI = 10 (min) should return 200."""
        res = client.post("/predict",
                          data=json.dumps({**VALID_INPUT, "BMI": 10}),
                          content_type="application/json")
        assert res.status_code == 200

    def test_bmi_at_maximum_boundary(self, client):
        """BMI = 100 (max) should return 200."""
        res = client.post("/predict",
                          data=json.dumps({**VALID_INPUT, "BMI": 100}),
                          content_type="application/json")
        assert res.status_code == 200

    def test_bmi_just_below_minimum(self, client):
        """BMI = 9.9 (just below min) should return 400."""
        res = client.post("/predict",
                          data=json.dumps({**VALID_INPUT, "BMI": 9.9}),
                          content_type="application/json")
        assert res.status_code == 400

    def test_age_at_minimum_boundary(self, client):
        """Age = 1 (min) should return 200."""
        res = client.post("/predict",
                          data=json.dumps({**VALID_INPUT, "Age": 1}),
                          content_type="application/json")
        assert res.status_code == 200

    def test_age_at_maximum_boundary(self, client):
        """Age = 13 (max) should return 200."""
        res = client.post("/predict",
                          data=json.dumps({**VALID_INPUT, "Age": 13}),
                          content_type="application/json")
        assert res.status_code == 200

    def test_menthlth_at_zero(self, client):
        """MentHlth = 0 (min) should return 200."""
        res = client.post("/predict",
                          data=json.dumps({**VALID_INPUT, "MentHlth": 0}),
                          content_type="application/json")
        assert res.status_code == 200

    def test_menthlth_at_thirty(self, client):
        """MentHlth = 30 (max) should return 200."""
        res = client.post("/predict",
                          data=json.dumps({**VALID_INPUT, "MentHlth": 30}),
                          content_type="application/json")
        assert res.status_code == 200
