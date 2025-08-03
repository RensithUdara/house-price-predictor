import utils.session
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pickle
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # For flash messages

# Load the model
with open('house_price_prediction.pkl','rb') as f:
    model = pickle.load(f)

# Feature information with descriptions and typical ranges
FEATURE_INFO = {
    'CRIM': {'description': 'Crime rate per capita by town', 'min': 0.0, 'max': 100.0, 'default': 0.3},
    'ZN': {'description': 'Proportion of residential land zoned for lots over 25,000 sq.ft.', 'min': 0.0, 'max': 100.0, 'default': 0.0},
    'INDUS': {'description': 'Proportion of non-retail business acres per town', 'min': 0.0, 'max': 30.0, 'default': 11.0},
    'CHAS': {'description': 'Charles River dummy variable (1 if tract bounds river; 0 otherwise)', 'min': 0, 'max': 1, 'default': 0},
    'NOX': {'description': 'Nitric oxides concentration (parts per 10 million)', 'min': 0.3, 'max': 1.0, 'default': 0.5},
    'RM': {'description': 'Average number of rooms per dwelling', 'min': 3.0, 'max': 10.0, 'default': 6.0},
    'AGE': {'description': 'Proportion of owner-occupied units built prior to 1940', 'min': 0.0, 'max': 100.0, 'default': 65.0},
    'DIS': {'description': 'Weighted distances to five Boston employment centres', 'min': 1.0, 'max': 15.0, 'default': 4.0},
    'RAD': {'description': 'Index of accessibility to radial highways', 'min': 1, 'max': 24, 'default': 5},
    'TAX': {'description': 'Full-value property-tax rate per $10,000', 'min': 150, 'max': 800, 'default': 300},
    'PTRATIO': {'description': 'Pupil-teacher ratio by town', 'min': 10.0, 'max': 25.0, 'default': 18.0},
    'B': {'description': 'Proportion of African Americans by town', 'min': 0.0, 'max': 400.0, 'default': 350.0},
    'LSTAT': {'description': '% lower status of the population', 'min': 1.0, 'max': 40.0, 'default': 12.0}
}

@app.route('/')
def home():
    return render_template('index.html', feature_info=FEATURE_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate and extract features with error handling
        features = []
        errors = []
        
        for feature_name in FEATURE_INFO.keys():
            value = request.form.get(feature_name, '').strip()
            
            if not value:
                errors.append(f"{feature_name} is required")
                continue
                
            try:
                float_value = float(value)
                feature_info = FEATURE_INFO[feature_name]
                
                # Validate range
                if float_value < feature_info['min'] or float_value > feature_info['max']:
                    errors.append(f"{feature_name} must be between {feature_info['min']} and {feature_info['max']}")
                    continue
                    
                features.append(float_value)
            except ValueError:
                errors.append(f"{feature_name} must be a valid number")
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('index.html', feature_info=FEATURE_INFO, form_data=request.form)
        
        # Make prediction
        features_array = np.array([features])
        prediction = model.predict(features_array)
        output = round(prediction[0] * 1000, 2)  # Convert to thousands for better readability
        
        # Save prediction history (in a real app, you'd use a database)
        prediction_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': dict(zip(FEATURE_INFO.keys(), features)),
            'prediction': output
        }
        
        flash(f'Prediction successful! Estimated price: ${output:,.2f}', 'success')
        return render_template('index.html', 
                             feature_info=FEATURE_INFO, 
                             prediction_text=f"Predicted House Price: ${output:,.2f}",
                             prediction_data=prediction_data)
        
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return render_template('index.html', feature_info=FEATURE_INFO, form_data=request.form)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        features = []
        
        for feature_name in FEATURE_INFO.keys():
            if feature_name not in data:
                return jsonify({'error': f'{feature_name} is required'}), 400
            features.append(float(data[feature_name]))
        
        features_array = np.array([features])
        prediction = model.predict(features_array)
        output = round(prediction[0] * 1000, 2)
        
        return jsonify({
            'prediction': output,
            'formatted_prediction': f"${output:,.2f}",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample')
def sample_data():
    """Provide sample data for testing"""
    sample = {
        'CRIM': 0.3,
        'ZN': 0.0,
        'INDUS': 11.0,
        'CHAS': 0,
        'NOX': 0.5,
        'RM': 6.0,
        'AGE': 65.0,
        'DIS': 4.0,
        'RAD': 5,
        'TAX': 300,
        'PTRATIO': 18.0,
        'B': 350.0,
        'LSTAT': 12.0
    }
    return render_template('index.html', feature_info=FEATURE_INFO, form_data=sample)

@app.route('/about')
def about():
    """Information about the model and features"""
    return render_template('about.html', feature_info=FEATURE_INFO)


if __name__ == "__main__":
    app.run(debug=True)