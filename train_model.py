import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Generate the data as previously described
np.random.seed(42)
num_samples = 10000
data = {
    'DistanceFromVenue': np.random.randint(1, 51, size=num_samples),
    'Availability': np.random.randint(0, 2, size=num_samples),
    'InterestLevel': np.random.randint(1, 11, size=num_samples),
    'RelationshipWithHost': np.random.randint(1, 11, size=num_samples),
    'SocialCircleAttendance': np.random.randint(0, 101, size=num_samples),
    'MaritalStatus': np.random.randint(0, 2, size=num_samples),  # Corrected name
    'PersonalInvitation': np.random.randint(0, 2, size=num_samples),
    'WeatherForecast': np.random.randint(0, 2, size=num_samples),
    'PreviousAttendance': np.random.randint(0, 2, size=num_samples),
    'TransportationAvailability': np.random.randint(0, 2, size=num_samples)
}

weights = {
    'DistanceFromVenue': -0.2,
    'Availability': 0.3,
    'InterestLevel': 0.4,
    'RelationshipWithHost': 0.3,
    'SocialCircleAttendance': 0.2,
    'MaritalStatus': 0.2,  # Corrected name
    'PersonalInvitation': 0.3,
    'WeatherForecast': 0.1,
    'PreviousAttendance': 0.3,
    'TransportationAvailability': 0.2
}

data['WillAttend'] = (
    (weights['DistanceFromVenue'] * data['DistanceFromVenue']) +
    (weights['Availability'] * data['Availability']) +
    (weights['InterestLevel'] * data['InterestLevel']) +
    (weights['RelationshipWithHost'] * data['RelationshipWithHost']) +
    (weights['SocialCircleAttendance'] * data['SocialCircleAttendance']) +
    (weights['MaritalStatus'] * data['MaritalStatus']) +  # Corrected name
    (weights['PersonalInvitation'] * data['PersonalInvitation']) +
    (weights['WeatherForecast'] * data['WeatherForecast']) +
    (weights['PreviousAttendance'] * data['PreviousAttendance']) +
    (weights['TransportationAvailability'] * data['TransportationAvailability'])
)

threshold = 3
data['WillAttend'] = (data['WillAttend'] > threshold).astype(int)

df = pd.DataFrame(data)

# Check the balance of the dataset
print(df['WillAttend'].value_counts())

# Training the logistic regression model
X = df.drop(columns=['WillAttend'])
y = df['WillAttend']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Saving the model and the scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler trained and saved as model.pkl and scaler.pkl")
