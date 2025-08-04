import os
import sagemaker
from common import (
    download_training_data,
    download_pretrained_model,
    package_and_upload_local_model,
    upload_data_to_s3,
    run_estimator_job,
    run_modeltrainer_job,
    deploy_local_model_endpoint,
    deploy_estimator_endpoint,
    deploy_modeltrainer_endpoint
)

def main():
    """Main execution function"""

    deployed_endpoints = []

    try:
        # Step 1: Download training data and pre-trained model
        download_training_data()
        download_pretrained_model()

        # Step 2: Initialize SageMaker session and get configuration
        sagemaker_session = sagemaker.Session()
        default_bucket = sagemaker_session.default_bucket()
        account_id = sagemaker_session.boto_session.client('sts').get_caller_identity()['Account']
        region = sagemaker_session.boto_region_name

        # Step 3: Upload data to S3
        train_s3_uri = upload_data_to_s3(sagemaker_session, default_bucket)

        # Step 4: Package and upload local model
        local_model_uri = package_and_upload_local_model(sagemaker_session, default_bucket)

        # Step 5: Set up configuration
        sagemaker_role = f"arn:aws:iam::{account_id}:role/SageMakerDefaultExecution"
        model_output_path = f"s3://{default_bucket}/models/california-housing/"

        # Step 6: Run Estimator training job
        sklearn_estimator = run_estimator_job(train_s3_uri, sagemaker_role, model_output_path, sagemaker_session)

        # Step 7: Run ModelTrainer training job
        model_trainer = run_modeltrainer_job(train_s3_uri, sagemaker_role, model_output_path, region)

        print("\n🎉 All training jobs completed successfully!")

        # Step 8: Deploy endpoints
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

        # Summary
        print("\n🎯 Deployment Summary:")
        print("=" * 60)
        for endpoint_type, name in deployed_endpoints:
            print(f"{endpoint_type:25} | {name}")
        print("=" * 60)
        print(f"Total endpoints deployed: {len(deployed_endpoints)}/3")

    except Exception as e:
        print(f"\n❌ Execution failed with error: {str(e)}")
        raise
    finally:
        # Cleanup temporary files
        for file in ['model.tar.gz']:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    main()