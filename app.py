from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model, scaler, PCA, and feature names
logging.info("Loading model, scaler, PCA, and feature names...")
with open('loan_default_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('pca.pkl', 'rb') as pca_file:
    pca = pickle.load(pca_file)
with open('feature_names.pkl', 'rb') as feature_file:
    feature_names = pickle.load(feature_file)
logging.info("Model, scaler, PCA, and feature names loaded successfully.")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        input_data = pd.DataFrame([data])
        
        # Convert input to match training features
        input_data = pd.get_dummies(input_data)
        missing_cols = set(feature_names) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0  # Add missing columns with default 0
        input_data = input_data[feature_names]  # Reorder columns
        
        logging.info("Processed input features: %s", input_data.columns.tolist())
        
        # Scale and apply PCA
        features = scaler.transform(input_data)
        features = pca.transform(features)
        
        prediction = model.predict(features)[0]
        result = "HE/SHE is Eligible(Default)" if prediction == 1 else "HE/SHE is NOt Eligible( Not Default)"
        
        logging.info("Prediction result: %s", result)
        
        return jsonify({'prediction': result})
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)