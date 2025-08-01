import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig

# Create a SageMaker session
sagemaker_session = sagemaker.Session()

# Retrieve the AWS account ID for constructing resource ARNs
account_id = sagemaker_session.account_id()

# Get the default S3 bucket
default_bucket = sagemaker_session.default_bucket()

# Define the SageMaker execution role
SAGEMAKER_ROLE = f"arn:aws:iam::{account_id}:role/SageMakerDefaultExecution"

# Set a name for the SageMaker Pipeline
PIPELINE_NAME = "california-housing-pipeline"

# Model package group name and endpoint name
MODEL_PACKAGE_GROUP_NAME = "california-housing-pipeline-models"
ENDPOINT_NAME = "california-housing-estimator"

# Step 1: Data Processing
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=SAGEMAKER_ROLE,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=sagemaker_session
)

processing_step = ProcessingStep(
    name="ProcessData",
    processor=processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=f"s3://{default_bucket}/datasets/california_housing_train.csv",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train_data",
            source="/opt/ml/processing/train"
        ),
        sagemaker.processing.ProcessingOutput(
            output_name="test_data",
            source="/opt/ml/processing/test"
        )
    ],
    code="pipeline/data_processing.py"
)

# Step 2: Model Training
estimator = SKLearn(
    entry_point="pipeline/train.py",
    role=SAGEMAKER_ROLE,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sagemaker_session
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri
        )
    }
)

# Step 3: Model Evaluation
evaluation_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=SAGEMAKER_ROLE,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=sagemaker_session
)

# Define property file for evaluation metrics
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=evaluation_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        sagemaker.processing.ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation"
        )
    ],
    code="pipeline/evaluation.py",
    property_files=[evaluation_report]
)

# Step 4: Conditional Model Registration with automatic approval
register_step = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
    approval_status="Approved"
)

# Define the quality threshold condition
condition_r2_threshold = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=evaluation_step.name,
        property_file=evaluation_report,
        json_path="regression_metrics.r2_score"
    ),
    right=0.6
)

# Create a conditional step that only registers the model if it meets quality standards
condition_step = ConditionStep(
    name="CheckModelQuality",
    conditions=[condition_r2_threshold],
    if_steps=[register_step],
    else_steps=[]
)

# Create pipeline by combining all steps
pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[processing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=sagemaker_session
)

try:
    # Create or update pipeline
    print("Creating/updating pipeline...")
    pipeline.upsert(role_arn=SAGEMAKER_ROLE)
    
    # Start pipeline execution
    print("Starting pipeline execution...")
    execution = pipeline.start()
    print(f"Pipeline execution ARN: {execution.arn}")
    
    # Wait for pipeline to complete
    print("Waiting for pipeline to complete...")
    execution.wait()
    
    # Get final status
    execution_details = execution.describe()
    status = execution_details['PipelineExecutionStatus']
    print(f"Pipeline execution completed with status: {status}")
    
    # Check if pipeline succeeded
    if status != 'Succeeded':
        print(f"Pipeline failed with status: {status}")
        exit(1)
    
    # Deploy the model if pipeline succeeded
    print("\nPipeline succeeded! Proceeding with deployment...")
    
    # Get the training job name from the pipeline execution
    steps = execution.list_steps()
    training_job_name = None
    
    for step in steps:
        if step.get('StepName') == 'TrainModel':
            metadata = step.get('Metadata', {})
            training_job = metadata.get('TrainingJob', {})
            training_job_name = training_job.get('Arn', '').split('/')[-1]
            break
    
    if not training_job_name:
        print("Could not find training job name from pipeline execution")
        exit(1)
    
    print(f"Found training job: {training_job_name}")
    
    # Get model artifacts from the completed training job
    training_job_details = sagemaker_session.describe_training_job(training_job_name)
    model_data = training_job_details['ModelArtifacts']['S3ModelArtifacts']

    # Create SKLearnModel with explicit configuration (same as working deployments)
    model = SKLearnModel(
        model_data=model_data,
        role=SAGEMAKER_ROLE,
        entry_point='pipeline/entry_point.py',
        framework_version='1.2-1',
        py_version='py3',
        sagemaker_session=sagemaker_session
    )
    
    # Configure serverless inference
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=5
    )
    
    # Deploy the model as a serverless endpoint
    print(f"Deploying model to serverless endpoint '{ENDPOINT_NAME}'...")
    predictor = model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=ENDPOINT_NAME,
        wait=True
    )
    
    print(f"\n✅ Pipeline execution and deployment completed successfully!")
    print(f"Model deployed to endpoint '{ENDPOINT_NAME}'")
    
except Exception as e:
    print(f"Error: {e}")