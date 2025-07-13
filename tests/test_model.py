import sys
import os

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.serve_model import predict, PatientData


def test_predict_returns_expected_keys_and_types():
    sample_input = PatientData(
        age=67,
        avg_glucose_level=105.92,
        bmi=36.6,
        gender="Female",
        ever_married="Yes",
        work_type="Private",
        Residence_type="Urban",
        smoking_status="formerly smoked",
    )
    result = predict(sample_input)

    assert "stroke_probability" in result
    assert "stroke_prediction" in result
    assert isinstance(result["stroke_probability"], float)
    assert isinstance(result["stroke_prediction"], int)
