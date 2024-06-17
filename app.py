from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the logistic regression model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = [data['DistanceFromVenue'], data['Availability'], data['InterestLevel'],
                data['RelationshipWithHost'], data['SocialCircleAttendance'], data['MaritalStatus'],
                data['PersonalInvitation'], data['WeatherForecast'], data['PreviousAttendance'],
                data['TransportationAvailability']]
    features = np.array(features, dtype=float).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    result = 'Will Attend' if prediction[0] == 1 else 'Will Not Attend'
    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
