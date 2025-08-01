import os
import argparse
import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def model_fn(model_dir):
    """Load model for inference"""
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args = parser.parse_args()
    
    # Load training data
    train_data_path = os.path.join(args.train, 'train.csv')
    df = pd.read_csv(train_data_path)
    
    # Separate features and target
    X_train = df.drop("MedHouseVal", axis=1)
    y_train = df["MedHouseVal"]
    
    print(f"Training with {len(X_train)} examples...")
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Save the model
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    print("Training completed!")