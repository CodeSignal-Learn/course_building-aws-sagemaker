import os
import joblib

def model_fn(model_dir):
    """Load model for inference"""
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model