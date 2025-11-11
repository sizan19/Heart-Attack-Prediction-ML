"""
Heart Disease Prediction Web Application
Built with Flask
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
label_encoder = model_data['label_encoder']
feature_names = model_data['feature_names']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        gender = int(request.form['gender'])
        heart_rate = float(request.form['heart_rate'])
        systolic_bp = float(request.form['systolic_bp'])
        diastolic_bp = float(request.form['diastolic_bp'])
        blood_sugar = float(request.form['blood_sugar'])
        ck_mb = float(request.form['ck_mb'])
        troponin = float(request.form['troponin'])

        # Prepare input data
        input_data = np.array([[age, gender, heart_rate, systolic_bp, diastolic_bp,
                                blood_sugar, ck_mb, troponin]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Get the label
        result_label = label_encoder.inverse_transform([prediction])[0]

        return jsonify({
            'prediction': result_label.upper(),
            'negative_probability': f"{prediction_proba[0] * 100:.2f}",
            'positive_probability': f"{prediction_proba[1] * 100:.2f}",
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
