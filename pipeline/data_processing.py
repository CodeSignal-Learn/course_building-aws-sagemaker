import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Read input data
    input_data_path = "/opt/ml/processing/input/california_housing.csv"
    df = pd.read_csv(input_data_path)

    print(f"Processing {len(df)} samples...")

    # 1. Cap all numeric features at their 95th percentiles to handle outliers
    # (excluding geographic coordinates which don't need capping)
    features_to_cap = df.select_dtypes(include=['float64']).columns.drop(['Latitude', 'Longitude'])

    for feature in features_to_cap:
        cap_value = df[feature].quantile(0.95)
        df[feature] = df[feature].clip(upper=cap_value)

    # 2. Create a new feature: average number of rooms per household
    df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']

    # Select relevant features for modeling
    feature_columns = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
        'AveOccup', 'Latitude', 'Longitude', 'RoomsPerHousehold'
    ]
    X = df[feature_columns]
    y = df['MedHouseVal']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create output directories
    os.makedirs("/opt/ml/processing/train", exist_ok=True)
    os.makedirs("/opt/ml/processing/test", exist_ok=True)

    # Combine features and target for saving
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Save training data
    train_data.to_csv("/opt/ml/processing/train/train.csv", index=False)

    # Save test data
    test_data.to_csv("/opt/ml/processing/test/test.csv", index=False)

    print("Data processing completed!")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")