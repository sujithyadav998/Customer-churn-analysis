from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('model.pkl', 'rb') as file:
    model, scaler = pickle.load(file)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/pred', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Ensure all required features are present
        required_features = [
            'Age', 
            'Tenure',
            'Usage Frequency',
            'Support Calls',
            'Payment Delay',
            'Total Spend',
            'Last Interaction'
        ]
        
        for feature in required_features:
            if feature not in input_data.columns:
                return jsonify({'error': f'Missing required feature: {feature}'}), 400
        
        # Ensure correct order of features
        input_data = input_data[required_features]
        
        # Scale the features using the loaded scaler
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        probability = model.predict_proba(input_data_scaled)
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),  # 0 for no churn, 1 for churn
            'churn_probability': float(probability[0][1])  # Probability of churning
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyse')
def analyse():
    return render_template('analyse.html')

if __name__ == '__main__':
    app.run(debug=True) 