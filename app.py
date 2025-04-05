from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and columns
model = joblib.load('model/house_price_model.pkl')
model_columns = pd.read_pickle('model/model_columns (1).pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        data = {
            'AREA': request.form['area'],
            'INT_SQFT': float(request.form['sqft']),
            'N_BEDROOM': int(request.form['bedrooms']),
            'N_BATHROOM': int(request.form['bathrooms']),
            'N_ROOM': int(request.form['rooms']),
            'SALE_COND': request.form['sale_cond'],
            'PARK_FACIL': request.form['parking'],
            'BUILDTYPE': request.form['buildtype'],
            'UTILITY_AVAIL': request.form['utility'],
            'STREET': request.form['street'],
            'MZZONE': request.form['mzzone'],
            'AGE': int(request.form['age'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # One-hot encode categorical features
        input_encoded = pd.get_dummies(input_df)
        
        # Align columns with training data
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]

        return render_template('index.html', 
                             prediction_text=f'Predicted Price: ₹{prediction:,.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)