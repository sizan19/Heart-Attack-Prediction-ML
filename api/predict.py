from http.server import BaseHTTPRequestHandler
import json
import pickle
import numpy as np
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
label_encoder = model_data['label_encoder']
feature_names = model_data['feature_names']


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Get content length
            content_length = int(self.headers.get('Content-Length', 0))
            
            # Read and parse the body
            body = self.rfile.read(content_length).decode('utf-8')
            
            # Parse form data or JSON
            if self.headers.get('Content-Type', '').startswith('application/x-www-form-urlencoded'):
                from urllib.parse import parse_qs
                data = parse_qs(body)
                age = float(data['age'][0])
                gender = int(data['gender'][0])
                heart_rate = float(data['heart_rate'][0])
                systolic_bp = float(data['systolic_bp'][0])
                diastolic_bp = float(data['diastolic_bp'][0])
                blood_sugar = float(data['blood_sugar'][0])
                ck_mb = float(data['ck_mb'][0])
                troponin = float(data['troponin'][0])
            else:
                data = json.loads(body)
                age = float(data['age'])
                gender = int(data['gender'])
                heart_rate = float(data['heart_rate'])
                systolic_bp = float(data['systolic_bp'])
                diastolic_bp = float(data['diastolic_bp'])
                blood_sugar = float(data['blood_sugar'])
                ck_mb = float(data['ck_mb'])
                troponin = float(data['troponin'])

            # Prepare input data
            input_data = np.array([[age, gender, heart_rate, systolic_bp, diastolic_bp,
                                    blood_sugar, ck_mb, troponin]])

            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            # Get the label
            result_label = label_encoder.inverse_transform([prediction])[0]

            response = {
                'prediction': result_label.upper(),
                'negative_probability': f"{prediction_proba[0] * 100:.2f}",
                'positive_probability': f"{prediction_proba[1] * 100:.2f}",
                'status': 'success'
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            response = {
                'status': 'error',
                'message': str(e)
            }
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
