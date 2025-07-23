import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def model_fn(model_dir):
    """Load model for inference - SageMaker will handle the rest"""
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

if __name__ == '__main__':
    # Parse SageMaker arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args = parser.parse_args()
    
    # Load the training data from S3
    train_data_path = os.path.join(args.train, 'california_housing_train.csv')
    df = pd.read_csv(train_data_path)
    
    # Separate features and target variable
    X_train = df.drop("MedHouseVal", axis=1)  # Features
    y_train = df["MedHouseVal"]               # Target: median house value
    
    print(f"Starting training with {len(X_train)} examples...")
    print(f"Features: {list(X_train.columns)}")
    
    # Create and train the model directly
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model performance on training data
    print(f"Starting evaluation...")
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Print training performance metrics
    print(f"R² Score: {train_r2:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    
    # Save the model for SageMaker deployment
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    print("Training completed!")