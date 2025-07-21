import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Create a directory named 'data'
os.makedirs("data", exist_ok=True)

# Load and convert the dataset to a DataFrame
data = fetch_california_housing(as_frame=True).frame

# Save the raw DataFrame as a CSV file
data.to_csv("data/california_housing.csv", index=False)

# Create a new feature: average number of rooms per household
data['RoomsPerHousehold'] = data['AveRooms'] / data['AveOccup']

# Select relevant features for modeling
feature_columns = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
    'AveOccup', 'Latitude', 'Longitude', 'RoomsPerHousehold'
]
X = data[feature_columns]
y = data['MedHouseVal']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cap numeric features at their 95th percentiles using ONLY training data
# (excluding geographic coordinates which don't need capping)
features_to_cap = X_train.select_dtypes(include=['float64']).columns.drop(['Latitude', 'Longitude'])

# Calculate capping thresholds from training data only
cap_values = {}
for feature in features_to_cap:
    cap_values[feature] = X_train[feature].quantile(0.95)

# Apply capping to both training and test sets using training-derived thresholds
for feature in features_to_cap:
    X_train[feature] = X_train[feature].clip(upper=cap_values[feature])
    X_test[feature] = X_test[feature].clip(upper=cap_values[feature])

# Combine features and target for saving
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the processed training and test datasets to CSV files
train_data.to_csv('data/california_housing_train.csv', index=False)
test_data.to_csv('data/california_housing_test.csv', index=False)