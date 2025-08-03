from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import json
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'sri_lanka_housing_predictor_2025'

# Sri Lanka specific features with LKR currency and local context
SRI_LANKA_FEATURE_INFO = {
    'LOCATION_SCORE': {'description': 'Location desirability score (1-10): Colombo=10, Gampaha=8, Kandy=7, etc.', 'min': 1, 'max': 10, 'default': 6},
    'LAND_SIZE': {'description': 'Land size in perches', 'min': 1, 'max': 500, 'default': 15},
    'HOUSE_SIZE': {'description': 'House size in square feet', 'min': 500, 'max': 10000, 'default': 2000},
    'BEDROOMS': {'description': 'Number of bedrooms', 'min': 1, 'max': 8, 'default': 3},
    'BATHROOMS': {'description': 'Number of bathrooms', 'min': 1, 'max': 6, 'default': 2},
    'PARKING': {'description': 'Parking spaces available', 'min': 0, 'max': 10, 'default': 2},
    'AGE_YEARS': {'description': 'Age of the house in years', 'min': 0, 'max': 100, 'default': 10},
    'AMENITIES_SCORE': {'description': 'Amenities score (1-10): Swimming pool, gym, garden, etc.', 'min': 1, 'max': 10, 'default': 5},
    'PROXIMITY_COLOMBO': {'description': 'Distance to Colombo in kilometers', 'min': 0, 'max': 300, 'default': 50},
    'SCHOOL_RATING': {'description': 'Nearby school quality rating (1-10)', 'min': 1, 'max': 10, 'default': 6},
    'TRANSPORT_ACCESS': {'description': 'Public transport accessibility (1-10)', 'min': 1, 'max': 10, 'default': 7},
    'COMMERCIAL_ACCESS': {'description': 'Access to shopping centers, banks, hospitals (1-10)', 'min': 1, 'max': 10, 'default': 6},
    'SECURITY_LEVEL': {'description': 'Area security level (1-10)', 'min': 1, 'max': 10, 'default': 7}
}

# Location mappings for Sri Lankan districts/cities
LOCATION_MAPPINGS = {
    'Colombo': 10, 'Mount Lavinia': 9, 'Dehiwala': 9, 'Moratuwa': 8.5,
    'Gampaha': 8, 'Negombo': 7.5, 'Kelaniya': 8, 'Kadawatha': 7.5,
    'Kandy': 7, 'Peradeniya': 6.5, 'Galle': 6, 'Matara': 5.5,
    'Jaffna': 5, 'Trincomalee': 4.5, 'Batticaloa': 4, 'Anuradhapura': 4,
    'Polonnaruwa': 3.5, 'Ratnapura': 4.5, 'Kegalle': 4.5, 'Kurunegala': 5,
    'Puttalam': 4, 'Hambantota': 4, 'Monaragala': 3, 'Badulla': 4.5,
    'Nuwara Eliya': 5.5
}

def generate_sri_lanka_sample_data():
    """Generate realistic sample data for Sri Lankan housing market"""
    np.random.seed(42)
    n_samples = 1000
    
    data = []
    for i in range(n_samples):
        # Generate correlated features that make sense for Sri Lankan market
        location_score = np.random.choice(list(LOCATION_MAPPINGS.values()))
        
        # Higher location scores tend to have smaller land but higher prices
        if location_score >= 9:  # Colombo area
            land_size = np.random.normal(10, 3)
            house_size = np.random.normal(1800, 400)
            base_price = 80000000  # 80M LKR base
        elif location_score >= 7:  # Gampaha, Kandy
            land_size = np.random.normal(15, 5)
            house_size = np.random.normal(2200, 500)
            base_price = 40000000  # 40M LKR base
        elif location_score >= 5:  # Other major cities
            land_size = np.random.normal(20, 8)
            house_size = np.random.normal(2000, 400)
            base_price = 20000000  # 20M LKR base
        else:  # Rural areas
            land_size = np.random.normal(30, 10)
            house_size = np.random.normal(1500, 300)
            base_price = 8000000   # 8M LKR base
        
        # Ensure minimum values
        land_size = max(1, land_size)
        house_size = max(500, house_size)
        
        bedrooms = max(1, int(house_size / 600))  # Approx 600 sqft per bedroom
        bathrooms = max(1, min(bedrooms, int(house_size / 800)))
        parking = np.random.randint(0, max(1, min(4, int(land_size / 5))))
        age_years = max(0, np.random.exponential(15))  # Most houses are relatively new
        amenities_score = np.random.randint(1, 11)
        proximity_colombo = max(0, np.random.normal(50, 30))
        school_rating = np.random.randint(1, 11)
        transport_access = max(1, min(10, location_score + np.random.randint(-2, 3)))
        commercial_access = max(1, min(10, location_score + np.random.randint(-2, 3)))
        security_level = max(1, min(10, location_score + np.random.randint(-1, 2)))
        
        # Calculate price based on features
        price = base_price
        price *= (house_size / 2000)  # Size factor
        price *= (land_size / 20)     # Land factor
        price *= (1 + (amenities_score - 5) * 0.1)  # Amenities factor
        price *= (1 - (age_years * 0.01))  # Age depreciation
        price *= (1 + (location_score - 5) * 0.15)  # Location premium
        price *= (1 - (proximity_colombo / 1000))  # Distance factor
        
        # Add some randomness
        price *= np.random.normal(1, 0.15)
        price = max(2000000, price)  # Minimum 2M LKR
        
        data.append({
            'LOCATION_SCORE': round(location_score, 1),
            'LAND_SIZE': round(land_size, 1),
            'HOUSE_SIZE': round(house_size),
            'BEDROOMS': int(bedrooms),
            'BATHROOMS': int(bathrooms),
            'PARKING': int(parking),
            'AGE_YEARS': round(age_years, 1),
            'AMENITIES_SCORE': int(amenities_score),
            'PROXIMITY_COLOMBO': round(proximity_colombo, 1),
            'SCHOOL_RATING': int(school_rating),
            'TRANSPORT_ACCESS': int(transport_access),
            'COMMERCIAL_ACCESS': int(commercial_access),
            'SECURITY_LEVEL': int(security_level),
            'PRICE_LKR': round(price)
        })
    
    return pd.DataFrame(data)

def train_sri_lanka_model():
    """Train a new model with Sri Lankan housing data"""
    print("Generating Sri Lankan housing dataset...")
    df = generate_sri_lanka_sample_data()
    
    # Prepare features and target
    feature_cols = list(SRI_LANKA_FEATURE_INFO.keys())
    X = df[feature_cols]
    y = df['PRICE_LKR']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model (better for housing data than Linear Regression)
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Absolute Error: LKR {mae:,.0f}")
    print(f"R² Score: {r2:.3f}")
    
    # Save model and scaler
    with open('sri_lanka_house_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    # Save sample data for reference
    df.to_csv('sri_lanka_housing_data.csv', index=False)
    
    return model, scaler

# Initialize or load the Sri Lankan model
model_path = 'sri_lanka_house_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
    print("Loaded existing Sri Lankan housing model")
else:
    print("Training new Sri Lankan housing model...")
    model, scaler = train_sri_lanka_model()

@app.route('/')
def home():
    return render_template('sl_index.html', feature_info=SRI_LANKA_FEATURE_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        errors = []
        
        for feature_name in SRI_LANKA_FEATURE_INFO.keys():
            value = request.form.get(feature_name, '').strip()
            
            if not value:
                errors.append(f"{feature_name} is required")
                continue
                
            try:
                float_value = float(value)
                feature_info = SRI_LANKA_FEATURE_INFO[feature_name]
                
                if float_value < feature_info['min'] or float_value > feature_info['max']:
                    errors.append(f"{feature_name} must be between {feature_info['min']} and {feature_info['max']}")
                    continue
                    
                features.append(float_value)
            except ValueError:
                errors.append(f"{feature_name} must be a valid number")
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('sl_index.html', feature_info=SRI_LANKA_FEATURE_INFO, form_data=request.form)
        
        # Make prediction
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)
        output = round(prediction[0])
        
        # Convert to different currencies for reference
        usd_rate = 300  # Approximate LKR to USD rate
        usd_price = output / usd_rate
        
        prediction_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': dict(zip(SRI_LANKA_FEATURE_INFO.keys(), features)),
            'prediction_lkr': output,
            'prediction_usd': round(usd_price)
        }
        
        flash(f'Prediction successful! Estimated price: LKR {output:,.0f} (USD {usd_price:,.0f})', 'success')
        return render_template('sl_index.html', 
                             feature_info=SRI_LANKA_FEATURE_INFO, 
                             prediction_text=f"Predicted House Price: LKR {output:,.0f}",
                             prediction_data=prediction_data)
        
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return render_template('sl_index.html', feature_info=SRI_LANKA_FEATURE_INFO, form_data=request.form)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        features = []
        
        for feature_name in SRI_LANKA_FEATURE_INFO.keys():
            if feature_name not in data:
                return jsonify({'error': f'{feature_name} is required'}), 400
            features.append(float(data[feature_name]))
        
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)
        output = round(prediction[0])
        
        usd_rate = 300
        usd_price = output / usd_rate
        
        return jsonify({
            'prediction_lkr': output,
            'prediction_usd': round(usd_price),
            'formatted_prediction': f"LKR {output:,.0f} (USD {usd_price:,.0f})",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample')
def sample_data():
    """Provide sample data for testing"""
    sample = {
        'LOCATION_SCORE': 8,  # Gampaha area
        'LAND_SIZE': 15,
        'HOUSE_SIZE': 2000,
        'BEDROOMS': 3,
        'BATHROOMS': 2,
        'PARKING': 2,
        'AGE_YEARS': 5,
        'AMENITIES_SCORE': 6,
        'PROXIMITY_COLOMBO': 25,
        'SCHOOL_RATING': 7,
        'TRANSPORT_ACCESS': 8,
        'COMMERCIAL_ACCESS': 7,
        'SECURITY_LEVEL': 8
    }
    return render_template('sl_index.html', feature_info=SRI_LANKA_FEATURE_INFO, form_data=sample)

@app.route('/about')
def about():
    """Information about the Sri Lankan model"""
    return render_template('sl_about.html', feature_info=SRI_LANKA_FEATURE_INFO, location_mappings=LOCATION_MAPPINGS)

@app.route('/compare')
def compare():
    """Compare different locations"""
    return render_template('sl_compare.html', feature_info=SRI_LANKA_FEATURE_INFO, location_mappings=LOCATION_MAPPINGS)

@app.route('/data-insights')
def data_insights():
    """Show market insights based on the generated data"""
    try:
        df = pd.read_csv('sri_lanka_housing_data.csv')
        
        insights = {
            'avg_price_by_location': df.groupby('LOCATION_SCORE')['PRICE_LKR'].mean().round().to_dict(),
            'avg_price_per_sqft': (df['PRICE_LKR'] / df['HOUSE_SIZE']).mean(),
            'most_expensive_area': df.loc[df['PRICE_LKR'].idxmax()],
            'cheapest_area': df.loc[df['PRICE_LKR'].idxmin()],
            'avg_house_size': df['HOUSE_SIZE'].mean(),
            'avg_land_size': df['LAND_SIZE'].mean()
        }
        
        return render_template('sl_insights.html', insights=insights)
    except Exception as e:
        flash(f'Could not load data insights: {str(e)}', 'error')
        return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Different port to avoid conflicts
