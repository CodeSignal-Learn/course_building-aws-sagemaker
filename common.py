import os
import requests
import sagemaker
import tarfile
import boto3
import json
import time
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, InputData, Compute, OutputDataConfig

# Constants
TRAINING_DATA_URL = "https://github.com/CodeSignal-Learn/course_building-aws-sagemaker/releases/latest/download/california_housing_train.csv"
PRETRAINED_MODEL_URL = "https://github.com/CodeSignal-Learn/course_building-aws-sagemaker/releases/latest/download/trained_model.joblib"

def create_sagemaker_role(role_name="SageMakerDefaultExecution"):
    """Create SageMaker execution role with necessary policies and verify it's assumable"""

    iam_client = boto3.client('iam')
    sts_client = boto3.client('sts')

    # Trust policy that allows SageMaker to assume this role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        account_id = sts_client.get_caller_identity()['Account']
        role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"

        # Check if role already exists
        role_exists = False
        try:
            iam_client.get_role(RoleName=role_name)
            print(f"✅ IAM role {role_name} already exists")
            role_exists = True
        except iam_client.exceptions.NoSuchEntityException:
            pass

        if not role_exists:
            # Create the role
            iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Default execution role for SageMaker training jobs and endpoints"
            )
            print(f"✅ Created IAM role: {role_name}")

            # Attach the necessary policies
            policies = [
                "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
                "arn:aws:iam::aws:policy/AmazonS3FullAccess"
            ]

            for policy_arn in policies:
                iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )
                print(f"✅ Attached policy {policy_arn} to role {role_name}")
            # Wait for role to be fully created
            time.sleep(15)  # Wait for role to propagate

        # Test role assumption with retries
        print("⏳ Testing role assumption...")
        max_retries = 120
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Check if the role exists
                iam_client.get_role(RoleName=role_name)
                print(f"✅ Role {role_name} exists!")
                return role_arn

            except Exception as assume_error:
                if attempt < max_retries - 1:
                    print(f"⏳ Role not ready yet (attempt {attempt + 1}/{max_retries}), waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ Role assumption failed after {max_retries} attempts: {assume_error}")
                    raise

        return role_arn

    except Exception as e:
        print(f"❌ Error creating SageMaker role: {e}")
        raise

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

def package_and_upload_local_model(sagemaker_session, default_bucket):
    """Package the local model and upload to S3"""

    print("Packaging local model...")

    # Create model.tar.gz
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add('models/trained_model.joblib', arcname='model.joblib')

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

def run_estimator_job(train_s3_uri, model_output_path, sagemaker_session):
    """Run training job using SKLearn Estimator"""

    print("Starting Estimator training job...")

    # Create and ensure SageMaker role is ready
    sagemaker_role = create_sagemaker_role()

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

def run_modeltrainer_job(train_s3_uri, model_output_path, region):
    """Run training job using ModelTrainer"""

    print("Starting ModelTrainer training job...")

    # Create and ensure SageMaker role is ready
    sagemaker_role = create_sagemaker_role()

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

def deploy_local_model_endpoint(local_model_uri, sagemaker_session):
    """Deploy the locally trained model to a serverless endpoint"""

    print("\n🚀 Deploying local model to serverless endpoint...")

    # Create and ensure SageMaker role is ready
    sagemaker_role = create_sagemaker_role()

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

def deploy_modeltrainer_endpoint(model_trainer, sagemaker_session):
    """Deploy the ModelTrainer model to a serverless endpoint"""

    print("\n🚀 Deploying ModelTrainer model to serverless endpoint...")

    # Create and ensure SageMaker role is ready
    sagemaker_role = create_sagemaker_role()

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
