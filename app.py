from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

#  MODELS 

# Diabetes
diabetes_model = pickle.load(open('models/diabetes_model.pkl','rb'))

# Heart
heart_model = pickle.load(open('models/heart_model.pkl','rb'))
heart_scaler = pickle.load(open('models/heart_scaler.pkl','rb'))
heart_columns = pickle.load(open('models/heart_columns.pkl','rb'))

# Parkinsons
parkinsons_model = pickle.load(open('models/parkinsons_model.pkl','rb'))
parkinsons_scaler = pickle.load(open('models/parkinsons_scaler.pkl','rb'))
parkinsons_columns = pickle.load(open('models/parkinsons_columns.pkl','rb'))

#  UTIL 

def get_risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

#  ROUTES 

@app.route('/')
def home():
    return render_template('dashboard.html')


#  DIABETES 

@app.route('/diabetes', methods=['GET','POST'])
def diabetes():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]

        columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                   'Insulin','BMI','DiabetesPedigreeFunction','Age']

        input_df = pd.DataFrame([features], columns=columns)

        prob = diabetes_model.predict_proba(input_df)[0][1]
        pred = diabetes_model.predict(input_df)[0]

        risk = get_risk_level(prob)

        return render_template(
            'diabetes.html',
            prediction=pred,
            probability=round(prob, 3),
            risk=risk
        )

    return render_template('diabetes.html')

#  HEART 

@app.route('/heart', methods=['GET','POST'])
def heart():
    if request.method == 'POST':
        form = request.form

        input_dict = {
            'Age': int(form['Age']),
            'Sex': form['Sex'],
            'ChestPainType': form['ChestPainType'],
            'RestingBP': int(form['RestingBP']),
            'Cholesterol': int(form['Cholesterol']),
            'FastingBS': int(form['FastingBS']),
            'RestingECG': form['RestingECG'],
            'MaxHR': int(form['MaxHR']),
            'ExerciseAngina': form['ExerciseAngina'],
            'Oldpeak': float(form['Oldpeak']),
            'ST_Slope': form['ST_Slope']
        }

        input_df = pd.DataFrame([input_dict])

        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=heart_columns, fill_value=0)

        numerical_cols = ['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']
        input_encoded[numerical_cols] = heart_scaler.transform(input_encoded[numerical_cols])

        prob = heart_model.predict_proba(input_encoded)[0][1]
        pred = heart_model.predict(input_encoded)[0]

        risk = get_risk_level(prob)

        return render_template(
            'heart.html',
            prediction=pred,
            probability=round(prob, 3),
            risk=risk
        )

    return render_template('heart.html')

#  PARKINSONS 

@app.route('/parkinsons', methods=['GET','POST'])
def parkinsons():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]

        input_df = pd.DataFrame([features], columns=parkinsons_columns)

        input_scaled = parkinsons_scaler.transform(input_df)

        prob = parkinsons_model.predict_proba(input_scaled)[0][1]
        pred = parkinsons_model.predict(input_scaled)[0]

        risk = get_risk_level(prob)

        return render_template(
            'parkinsons.html',
            prediction=pred,
            probability=round(prob, 3),
            risk=risk
        )

    return render_template('parkinsons.html')


if __name__ == "__main__":
    app.run(debug=True)
