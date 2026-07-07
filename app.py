from flask import Flask, request, render_template, flash, redirect, url_for
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Flask needs a secret key to use flash messages
app.secret_key = 'my_super_secret_key_123' 

# Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score')),
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Store the result in a flash message and redirect to the GET page
        flash(f"Predicted Math Score: {results[0]}", "result")
        return redirect(url_for('predict_datapoint'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)