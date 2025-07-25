import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create a directory named 'data'
os.makedirs("models", exist_ok=True)

# Load training data
train_data = pd.read_csv('data/california_housing_train.csv')
X_train = train_data.drop('MedHouseVal', axis=1)
y_train = train_data['MedHouseVal']

# Create and train the model directly
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/trained_model.joblib')