import os
import requests
import sagemaker
import tarfile
import shutil
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, InputData, Compute, OutputDataConfig

# Constants
TRAINING_DATA_URL = "https://github.com/CodeSignal-Learn/course_building-aws-sagemaker/releases/latest/download/california_housing_train.csv"
PRETRAINED_MODEL_URL = "https://github.com/CodeSignal-Learn/course_building-aws-sagemaker/releases/latest/download/trained_model.joblib"

def download_training_data():
    """Download california_housing_train.csv from GitHub release"""
    
    os.makedirs("data", exist_ok=True)
    
    print("Downloading training data...")
    response = requests.get(TRAINING_DATA_URL)
    response.raise_for_status()
    
    with open("data/california_housing_train.csv", 'wb') as f:
        f.write(response.content)
    
    print("✅ Training data downloaded successfully!")

def download_pretrained_model():
    """Download pretrained model from GitHub release"""
    
    os.makedirs("models", exist_ok=True)
    
    print("Downloading pretrained model...")
    response = requests.get(PRETRAINED_MODEL_URL)
    response.raise_for_status()
    
    with open("models/trained_model.joblib", 'wb') as f:
        f.write(response.content)
    
    print("✅ Pretrained model downloaded successfully!")

def create_entry_point_script():
    """Create entry_point.py for model inference"""
    
    entry_point_content = """import os
import joblib

def model_fn(model_dir):
    \"\"\"Load model for inference\"\"\"
    # Try both possible filenames
    model_path = os.path.join(model_dir, 'model.joblib')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'trained_model.joblib')
    
    model = joblib.load(model_path)
    return model
"""
    
    with open("entry_point.py", "w") as f:
        f.write(entry_point_content)
    
    print("✅ Entry point script created!")

def package_and_upload_local_model(sagemaker_session, default_bucket):
    """Package the local model and upload to S3"""
    
    print("Packaging local model...")
    
    # Create model.tar.gz
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add('models/trained_model.joblib', arcname='trained_model.joblib')
    
    # Upload to S3
    model_artifact_uri = sagemaker_session.upload_data(
        path='model.tar.gz',
        bucket=default_bucket,
        key_prefix='models/local-trained'
    )
    
    print(f"✅ Local model uploaded to: {model_artifact_uri}")
    return model_artifact_uri

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

def deploy_local_model_endpoint(local_model_uri, sagemaker_role, sagemaker_session):
    """Deploy the locally trained model to a serverless endpoint"""
    
    print("\n🚀 Deploying local model to serverless endpoint...")
    
    endpoint_name = "california-housing-local-model"
    
    try:
        # Create SKLearnModel from local model artifacts
        model = SKLearnModel(
            model_data=local_model_uri,
            role=sagemaker_role,
            entry_point='entry_point.py',
            framework_version='1.2-1',
            py_version='py3',
            sagemaker_session=sagemaker_session
        )
        
        # Configure serverless inference
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=2048,
            max_concurrency=10
        )
        
        # Deploy model
        predictor = model.deploy(
            serverless_inference_config=serverless_config,
            endpoint_name=endpoint_name,
            wait=True
        )
        
        print(f"✅ Local model endpoint deployed: {endpoint_name}")
        return endpoint_name
        
    except Exception as e:
        print(f"❌ Error deploying local model: {e}")
        return None

def deploy_estimator_endpoint(estimator):
    """Deploy the estimator model to a serverless endpoint"""
    
    print("\n🚀 Deploying estimator model to serverless endpoint...")
    
    endpoint_name = "california-housing-estimator"
    
    try:
        # Configure serverless inference
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=2048,
            max_concurrency=10
        )
        
        # Deploy estimator
        predictor = estimator.deploy(
            serverless_inference_config=serverless_config,
            endpoint_name=endpoint_name,
            wait=True
        )
        
        print(f"✅ Estimator endpoint deployed: {endpoint_name}")
        return endpoint_name
        
    except Exception as e:
        print(f"❌ Error deploying estimator: {e}")
        return None

def deploy_modeltrainer_endpoint(model_trainer, sagemaker_role, sagemaker_session):
    """Deploy the ModelTrainer model to a serverless endpoint"""
    
    print("\n🚀 Deploying ModelTrainer model to serverless endpoint...")
    
    endpoint_name = "california-housing-modeltrainer"
    
    try:
        # Get the training job name and retrieve model artifacts
        training_job_name = model_trainer._latest_training_job.training_job_name
        training_job_details = sagemaker_session.describe_training_job(training_job_name)
        model_data = training_job_details['ModelArtifacts']['S3ModelArtifacts']
        
        # Create SKLearnModel
        model = SKLearnModel(
            model_data=model_data,
            role=sagemaker_role,
            entry_point='entry_point.py',
            framework_version='1.2-1',
            py_version='py3',
            sagemaker_session=sagemaker_session
        )
        
        # Configure serverless inference
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=2048,
            max_concurrency=10
        )
        
        # Deploy model
        predictor = model.deploy(
            serverless_inference_config=serverless_config,
            endpoint_name=endpoint_name,
            wait=True
        )
        
        print(f"✅ ModelTrainer endpoint deployed: {endpoint_name}")
        return endpoint_name
        
    except Exception as e:
        print(f"❌ Error deploying ModelTrainer model: {e}")
        return None

def deploy_realtime_endpoint(estimator):
    """Deploy a real-time endpoint for high-throughput inference"""
    
    print("\n🚀 Deploying real-time endpoint...")
    
    endpoint_name = "california-housing-realtime"
    
    try:
        # Deploy as real-time endpoint
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.large',
            endpoint_name=endpoint_name,
            wait=True
        )
        
        print(f"✅ Real-time endpoint deployed: {endpoint_name}")
        return endpoint_name
        
    except Exception as e:
        print(f"❌ Error deploying real-time endpoint: {e}")
        return None

def main():
    """Main execution function"""
    
    deployed_endpoints = []
    
    try:
        # Step 1: Download training data
        download_training_data()

        # Step 2: Create entry point script
        create_entry_point_script()

        # Step 3: Initialize SageMaker session and get configuration
        sagemaker_session = sagemaker.Session()
        default_bucket = sagemaker_session.default_bucket()
        account_id = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
        region = sagemaker_session.boto_region_name

        # Step 4: Upload data to S3
        train_s3_uri = upload_data_to_s3(sagemaker_session, default_bucket)
        
        # Step 5: Package and upload local model
        local_model_uri = package_and_upload_local_model(sagemaker_session, default_bucket)

        # Step 6: Set up configuration
        sagemaker_role = f"arn:aws:iam::{account_id}:role/SageMakerDefaultExecution"
        model_output_path = f"s3://{default_bucket}/models/california-housing/"

        # Step 7: Run Estimator training job
        sklearn_estimator = run_estimator_job(train_s3_uri, sagemaker_role, model_output_path, sagemaker_session)

        # Step 8: Run ModelTrainer training job
        model_trainer = run_modeltrainer_job(train_s3_uri, sagemaker_role, model_output_path, region)

        print("\n🎉 All training jobs completed successfully!")
        
        # Step 9: Deploy endpoints
        print("\n📡 Starting endpoint deployments...")
        
        # Deploy local model endpoint
        endpoint_name = deploy_local_model_endpoint(local_model_uri, sagemaker_role, sagemaker_session)
        if endpoint_name:
            deployed_endpoints.append(("Local Model Serverless", endpoint_name))
        
        # Deploy estimator endpoint
        endpoint_name = deploy_estimator_endpoint(sklearn_estimator)
        if endpoint_name:
            deployed_endpoints.append(("Estimator Serverless", endpoint_name))
        
        # Deploy ModelTrainer endpoint
        endpoint_name = deploy_modeltrainer_endpoint(model_trainer, sagemaker_role, sagemaker_session)
        if endpoint_name:
            deployed_endpoints.append(("ModelTrainer Serverless", endpoint_name))
        
        # Deploy real-time endpoint
        endpoint_name = deploy_realtime_endpoint(sklearn_estimator)
        if endpoint_name:
            deployed_endpoints.append(("Real-time", endpoint_name))
        
        # Summary
        print("\n🎯 Deployment Summary:")
        print("=" * 60)
        for endpoint_type, name in deployed_endpoints:
            print(f"{endpoint_type:25} | {name}")
        print("=" * 60)
        print(f"Total endpoints deployed: {len(deployed_endpoints)}/4")
        
    except Exception as e:
        print(f"\n❌ Execution failed with error: {str(e)}")
        raise
    finally:
        # Cleanup temporary files
        for file in ['model.tar.gz', 'entry_point.py']:
            if os.path.exists(file):
                os.remove(file)
        
        # Cleanup models directory
        if os.path.exists('models'):
            shutil.rmtree('models')

if __name__ == "__main__":
    main()