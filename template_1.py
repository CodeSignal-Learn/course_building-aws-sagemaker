import os
import requests
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, InputData, Compute, OutputDataConfig

# Download training data from GitHub release
def download_training_data():
    """Download california_housing_train.csv from GitHub release"""
    
    os.makedirs("data", exist_ok=True)
    
    url = "https://github.com/CodeSignal-Learn/course_building-aws-sagemaker/releases/download/v2/california_housing_train.csv"
    
    print("Downloading training data...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open("data/california_housing_train.csv", 'wb') as f:
        f.write(response.content)
    
    print("✅ Training data downloaded successfully!")

# Download the training data
download_training_data()

# Initialize session and get configuration
sagemaker_session = sagemaker.Session()
default_bucket = sagemaker_session.default_bucket()
account_id = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
region = sagemaker_session.boto_region_name

# Upload data to S3
train_s3_uri = sagemaker_session.upload_data(
    path="data/california_housing_train.csv",
    bucket=default_bucket,
    key_prefix="datasets"
)

# Configuration constants
SAGEMAKER_ROLE = f"arn:aws:iam::{account_id}:role/SageMakerDefaultExecution"
MODEL_OUTPUT_PATH = f"s3://{default_bucket}/models/california-housing/"
INSTANCE_TYPE = "ml.m5.large"
INSTANCE_COUNT = 1

# Train with Estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=SAGEMAKER_ROLE,
    instance_type=INSTANCE_TYPE,
    instance_count=INSTANCE_COUNT,
    framework_version='1.2-1',
    py_version='py3',
    script_mode=True,
    sagemaker_session=sagemaker_session,
    output_path=MODEL_OUTPUT_PATH
)
sklearn_estimator.fit({'train': train_s3_uri}, wait=False)

# Train with ModelTrainer
sklearn_image = sagemaker.image_uris.retrieve(
    framework="sklearn",
    region=region,
    version="1.2-1",
    py_version="py3",
    instance_type=INSTANCE_TYPE
)

model_trainer = ModelTrainer(
    training_image=sklearn_image,
    source_code=SourceCode(source_dir=".", entry_script="train.py"),
    base_job_name="sklearn-modeltrainer",
    role=SAGEMAKER_ROLE,
    compute=Compute(instance_type=INSTANCE_TYPE, instance_count=INSTANCE_COUNT, volume_size_in_gb=30),
    output_data_config=OutputDataConfig(s3_output_path=MODEL_OUTPUT_PATH)
)

model_trainer.train(
    input_data_config=[InputData(channel_name="train", data_source=train_s3_uri)], 
    wait=False
)