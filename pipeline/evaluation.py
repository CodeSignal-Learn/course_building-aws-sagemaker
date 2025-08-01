import os
import json
import tarfile
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

if __name__ == "__main__":
    # Extract model artifacts
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, 'r:gz') as tar:
        tar.extractall("/opt/ml/processing/model/")
    
    # Load the trained model
    model = joblib.load("/opt/ml/processing/model/model.joblib")
    
    # Load test data
    test_data_path = "/opt/ml/processing/test/test.csv"
    test_df = pd.read_csv(test_data_path)
    
    # Separate features and target
    X_test = test_df.drop("MedHouseVal", axis=1)
    y_test = test_df["MedHouseVal"]
    
    print(f"Evaluating model on {len(X_test)} test samples...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create evaluation report
    evaluation_report = {
        "regression_metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2)
        }
    }
    
    print(f"Evaluation Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save evaluation report
    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/evaluation.json", "w") as f:
        json.dump(evaluation_report, f)
    
    print("Evaluation completed!")