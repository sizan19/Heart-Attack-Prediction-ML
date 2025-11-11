# Heart Disease Prediction Web Application

A machine learning-powered web application to predict heart disease risk based on clinical parameters.

## Features

- Interactive web interface built with Flask
- Real-time heart disease risk prediction
- Random Forest Classifier with 97.35% accuracy
- User-friendly input forms for clinical data
- Visual risk assessment with probability scores
- Beautiful and responsive UI design

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Train the model (one-time setup)**:
```bash
python train_model.py
```
This will create `model.pkl` file containing the trained Random Forest model with 97.35% accuracy.

2. **Run the web application**:
```bash
python flask_app.py
```

3. Open your browser and navigate to:
   - Local: `http://127.0.0.1:5000`
   - Network: `http://192.168.10.81:5000` (accessible from other devices on your network)

## Input Parameters

The application requires the following patient information:

- **Age**: Patient's age in years
- **Gender**: Male or Female
- **Heart Rate**: Beats per minute (bpm)
- **Systolic Blood Pressure**: Upper blood pressure reading (mmHg)
- **Diastolic Blood Pressure**: Lower blood pressure reading (mmHg)
- **Blood Sugar**: Blood glucose level (mg/dL)
- **CK-MB**: Creatine Kinase-MB cardiac marker (ng/mL)
- **Troponin**: Cardiac troponin level (ng/mL)

## Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 97.35%
- **F1 Score**: 97.86%

## Project Structure

```
heartdiseaseprediction/
│
├── notebook and datasets/
│   ├── heart-attack-preddiction.ipynb
│   └── Medicaldataset.csv
│
├── templates/
│   └── index.html         # Web interface HTML
│
├── flask_app.py           # Flask web application
├── train_model.py         # Model training script
├── model.pkl              # Trained model (generated)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Important Notes

- This is a predictive tool and should NOT replace professional medical advice
- Always consult with healthcare professionals for proper diagnosis and treatment
- The model is trained on historical data and predictions should be validated clinically

## License

This project is for educational and research purposes only.
