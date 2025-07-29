import sagemaker
from common import (
    download_training_data,
    upload_data_to_s3,
    run_estimator_job,
    run_modeltrainer_job
)

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