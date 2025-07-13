from src.serve_model import predict, PatientData


def test_predict():
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
