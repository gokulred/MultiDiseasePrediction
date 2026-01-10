from flask import Flask, render_template, request
import pickle
import numpy as np 
import pandas as pd 

app = Flask(__name__)


model = pickle.load(open('models/diabetes_model.pkl','rb'))
scaler = pickle.load(open('models/diabetes_scaler.pkl','rb'))

def get_risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes', methods=['GET','POST'])
def diabetes():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
           'Insulin','BMI','DiabetesPedigreeFunction','Age']
         
        final_input = pd.DataFrame([features],columns = columns)

        final_input = scaler.transform(final_input)

        prob = model.predict_proba(final_input)[0][1]
        pred = model.predict(final_input)[0]

        risk = get_risk_level(prob)

        return render_template(
            'diabetes.html',
            prediction=pred,
            probability=round(prob, 3),
            risk=risk
        )

    return render_template('diabetes.html')

if __name__ == "__main__":
    app.run(debug=True)
