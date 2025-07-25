import os
import requests
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, InputData, Compute, OutputDataConfig

# Constants
TRAINING_DATA_URL = "https://github.com/CodeSignal-Learn/course_building-aws-sagemaker/releases/latest/download/california_housing_train.csv"

def download_training_data():
    """Download california_housing_train.csv from GitHub release"""
    
    os.makedirs("data", exist_ok=True)
    
    print("Downloading training data...")
    response = requests.get(TRAINING_DATA_URL)
    response.raise_for_status()
    
    with open("data/california_housing_train.csv", 'wb') as f:
        f.write(response.content)
    
    print("✅ Training data downloaded successfully!")

def upload_data_to_s3(sagemaker_session, default_bucket, data_path="data/california_housing_train.csv"):
    """Upload training data to S3"""
    
    print("Uploading data to S3...")
    train_s3_uri = sagemaker_session.upload_data(
        path=data_path,
        bucket=default_bucket,
        key_prefix="datasets"
    )
    
    print(f"✅ Data uploaded to: {train_s3_uri}")
    return train_s3_uri

def run_estimator_job(train_s3_uri, sagemaker_role, model_output_path, sagemaker_session):
    """Run training job using SKLearn Estimator"""
    
    print("Starting Estimator training job...")
    
    # Configuration constants
    INSTANCE_TYPE = "ml.m5.large"
    INSTANCE_COUNT = 1
    
    sklearn_estimator = SKLearn(
        entry_point='train.py',
        role=sagemaker_role,
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        framework_version='1.2-1',
        py_version='py3',
        script_mode=True,
        sagemaker_session=sagemaker_session,
        output_path=model_output_path
    )
    
    sklearn_estimator.fit({'train': train_s3_uri}, wait=True)
    print("✅ Estimator training job completed!")
    
    return sklearn_estimator

def run_modeltrainer_job(train_s3_uri, sagemaker_role, model_output_path, region):
    """Run training job using ModelTrainer"""
    
    print("Starting ModelTrainer training job...")
    
    # Configuration constants
    INSTANCE_TYPE = "ml.m5.large"
    INSTANCE_COUNT = 1
    
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
        role=sagemaker_role,
        compute=Compute(instance_type=INSTANCE_TYPE, instance_count=INSTANCE_COUNT, volume_size_in_gb=30),
        output_data_config=OutputDataConfig(s3_output_path=model_output_path)
    )

    model_trainer.train(
        input_data_config=[InputData(channel_name="train", data_source=train_s3_uri)], 
        wait=True
    )
    
    print("✅ ModelTrainer training job completed!")
    return model_trainer

def main():
    """Main execution function"""
    
    try:
        # Step 1: Download training data
        download_training_data()

        # Step 2: Initialize SageMaker session and get configuration
        sagemaker_session = sagemaker.Session()
        default_bucket = sagemaker_session.default_bucket()
        account_id = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
        region = sagemaker_session.boto_region_name

        # Step 3: Upload data to S3
        train_s3_uri = upload_data_to_s3(sagemaker_session, default_bucket)

        # Step 4: Set up configuration
        sagemaker_role = f"arn:aws:iam::{account_id}:role/SageMakerDefaultExecution"
        model_output_path = f"s3://{default_bucket}/models/california-housing/"

        # Step 5: Run Estimator training job
        sklearn_estimator = run_estimator_job(train_s3_uri, sagemaker_role, model_output_path, sagemaker_session)

        # Step 6: Run ModelTrainer training job
        model_trainer = run_modeltrainer_job(train_s3_uri, sagemaker_role, model_output_path, region)

        print("🎉 All training jobs completed successfully!")
    except Exception as e:
        print(f"\n❌ Execution failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()