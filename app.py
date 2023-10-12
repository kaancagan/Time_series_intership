from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

app = Flask(__name__)

# Load your DataFrame 'df' here
df = pd.read_csv('/Users/ayhancagan/Desktop/tıme series:internship/data.csv')  # Replace with your actual data file

# Assuming your target column is named 'orders'
target_series = df['orders']

# Exogenous variables
exog_variables = df[['temperature', 'media_spend']]  # Adjust as needed

# Load the fitted model
fitted_model_filename = '/Users/ayhancagan/Desktop/tıme series:internship/fitted_sarimax_model.joblib'
loaded_model = joblib.load(fitted_model_filename)

# Home page with input form
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        start_date = pd.to_datetime(request.form['start_date'])
        temperature = float(request.form['temperature'])
        num_predictions = int(request.form['num_predictions'])
        
        exog_values = [temperature, 0]  # Assume media_spend is 0 for simplicity
        
        predictions = []

        for _ in range(num_predictions):
            forecast = loaded_model.get_forecast(steps=1, exog=[exog_values])
            predicted_value = round(forecast.predicted_mean.tolist()[0])  # Round the predicted value
            predictions.append(predicted_value)

            # Update the exogenous values for the next prediction
            start_date += pd.DateOffset(days=1)
            exog_values = [temperature, 0]  # Update other exogenous variables as needed
        
        # Redirect to the result page with the predictions
        return redirect(url_for('result', predictions=predictions))
    else:
        return render_template('index.html', submitted=False)

# Result page to display predictions
@app.route('/result')
def result():
    predictions = request.args.getlist('predictions')
    return render_template('result.html', predictions=predictions)

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5001)
