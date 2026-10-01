import sys
import os
import unittest
from fastapi.testclient import TestClient

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.main import app

class TestClinicalAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["module"], "clinical")
        self.assertEqual(data["models"]["heart"], "loaded")
        self.assertEqual(data["models"]["stroke"], "loaded")

    def test_predict_heart_valid(self):
        payload = {
            "age": 52,
            "sex": 1,
            "cp": 0,
            "trestbps": 125,
            "chol": 212,
            "fbs": 0,
            "restecg": 1,
            "thalach": 168,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 2,
            "ca": 2,
            "thal": 3
        }
        response = self.client.post("/predict/heart", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["module"], "heart")
        self.assertIn("prediction", data)
        self.assertIn("probability", data)
        self.assertIn("important_features", data)
        self.assertEqual(len(data["important_features"]), 4)
        self.assertIn("disclaimer", data)
        self.assertTrue("does not provide a medical diagnosis" in data["disclaimer"])

    def test_predict_stroke_valid(self):
        payload = {
            "gender": "Male",
            "age": 67.0,
            "hypertension": 0,
            "heart_disease": 1,
            "ever_married": "Yes",
            "work_type": "Private",
            "Residence_type": "Urban",
            "avg_glucose_level": 228.69,
            "bmi": 36.6,
            "smoking_status": "formerly smoked"
        }
        response = self.client.post("/predict/stroke", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["module"], "stroke")
        self.assertIn("prediction", data)
        self.assertIn("probability", data)
        self.assertIn("important_features", data)
        self.assertEqual(len(data["important_features"]), 4)
        self.assertIn("disclaimer", data)

    def test_predict_stroke_missing_bmi(self):
        # BMI is optional, should trigger median imputation cleanly
        payload = {
            "gender": "Female",
            "age": 61.0,
            "hypertension": 0,
            "heart_disease": 0,
            "ever_married": "Yes",
            "work_type": "Self-employed",
            "Residence_type": "Rural",
            "avg_glucose_level": 202.21,
            "bmi": None,
            "smoking_status": "never smoked"
        }
        response = self.client.post("/predict/stroke", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["module"], "stroke")
        self.assertIn("prediction", data)

    def test_predict_heart_invalid_cholesterol(self):
        # Invalid cholesterol > 700
        payload = {
            "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 9999,
            "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0,
            "slope": 2, "ca": 2, "thal": 3
        }
        response = self.client.post("/predict/heart", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error"], "Validation Error")

    def test_predict_stroke_invalid_gender(self):
        # Invalid gender string
        payload = {
            "gender": "Robot",
            "age": 67.0, "hypertension": 0, "heart_disease": 1,
            "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
            "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "smokes"
        }
        response = self.client.post("/predict/stroke", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error"], "Validation Error")

if __name__ == "__main__":
    unittest.main()
