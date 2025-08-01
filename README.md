# Enterprise Machine Learning with Amazon SageMaker

A comprehensive repository containing templates, scripts, and resources for the **"Enterprise Machine Learning with Amazon SageMaker"** learning path. This repository provides hands-on materials for setting up AWS SageMaker training jobs, pipelines and deploying machine learning endpoints across multiple courses.

## 🎯 Purpose

This repository is designed to set up resources for AWS accounts used in the SageMaker learning path. It includes:

- **Training Templates**: Launch SageMaker training jobs using different approaches
- **Deployment Templates**: Deploy models to both serverless and real-time endpoints  
- **Pipeline Templates**: Creates SageMaker pipelines and deploy approved model
- **Supporting Scripts**: Training and inference scripts for the California Housing dataset
- **Utility Functions**: Helper scripts for data processing and model management
- **Automated Data Pipeline**: GitHub Actions for processing and distributing datasets

## 🏗 Repository Structure

### 📋 Templates

| Template | Purpose | Used In | Features |
|----------|---------|---------|----------|
| `sagemakerTraining.py` | Launches Estimator + ModelTrainer training jobs | Course 2 | Training job creation and management |
| `sagemakerDeployServerless.py` | Training jobs + 3 serverless endpoints | Course 3 | Serverless inference deployment |
| `sagemakerDeployRealTime.py` | All above + real-time endpoint | Course 3 | Real-time inference with persistent instances |
| `sagemakerPipeline.py` | Creates 4 ML pipeline templates + serverless endpoint | Course 4 | Pipeline orchestration |

### 🔧 Template Components

- **`common.py`**: Shared helper functions for templates including:
  - Data download and S3 upload utilities
  - Training job orchestration (Estimator & ModelTrainer)
  - Endpoint deployment functions (serverless & real-time)
  - Model packaging and artifact management

- **`train.py`**: SageMaker training script for the California Housing dataset
- **`entry_point.py`**: Model inference entry point for deployed endpoints

- **`pipeline/`**: Folder with scripts used by the pipeline template

### 📁 Data & Models

- **`data/`**: Training datasets (generated via GitHub Actions)
  - `california_housing_train.csv` - Training set (16,512 rows)
  - `california_housing_test.csv` - Test set (4,128 rows)

- **`models/`**: Pre-trained model artifacts
  - `trained_model.joblib` - Ready-to-use LinearRegression model

### 🛠 Utilities

- **`utils/download_and_preprocess_datasets.py`**: Downloads and preprocesses the original California Housing dataset from scikit-learn
- **`utils/train_local_model.py`**: Trains and saves a model locally for testing

### 📦 Dependencies

- **`template_requirements.txt`**: Minimal dependencies needed to run the templates

- **`path_requirements.txt`**: Complete dependencies for the entire learning path

## 🚀 Quick Start

### Prerequisites
- AWS Account with SageMaker access
- IAM Role: `SageMakerDefaultExecution` 
- Python 3.10+

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd course_building-aws-sagemaker

# Install dependencies
pip install -r template_requirements.txt

# For complete learning path
pip install -r path_requirements.txt
```

### Usage

#### Course 2: Training Jobs
```bash
python sagemakerTraining.py
```
- Downloads California Housing dataset
- Uploads data to S3
- Launches Estimator training job
- Launches ModelTrainer training job

#### Course 3: Serverless Deployment  
```bash
python sagemakerDeployServerless.py
```
- Runs all training jobs
- Deploys 3 serverless endpoints:
  - Local pre-trained model endpoint
  - Estimator model endpoint  
  - ModelTrainer model endpoint

#### Course 3: Real-time Deployment
```bash
python sagemakerDeployRealTime.py
```
- Runs all training and serverless deployments
- Deploys additional real-time endpoint with `ml.m5.large` instance

#### Course 4: SageMaker Pipelines
```bash
python sagemakerPipeline.py
```
- Creates 4 pipelines with progressive complexity
- Executes all pipelines in parallel for maximum efficiency  
- Deploys erverless endpoint from conditional pipeline

## 🔄 Automated Data Pipeline

The repository uses **GitHub Actions** to automatically generate and distribute datasets:

- **Trigger**: Changes to preprocessing scripts or manual workflow dispatch
- **Process**: Downloads original dataset, applies preprocessing, creates train/test splits
- **Output**: Releases processed datasets and pre-trained models
- **Distribution**: Available via GitHub Releases and GitHub Pages

### Data Access URLs

**📄 GitHub Pages**: [https://codesignal-learn.github.io/course_building-aws-sagemaker/](https://codesignal-learn.github.io/course_building-aws-sagemaker/)

```bash
# Training data
wget https://github.com/CodeSignal-Learn/course_building-aws-sagemaker/releases/latest/download/california_housing_train.csv

# Pre-trained model  
wget https://github.com/CodeSignal-Learn/course_building-aws-sagemaker/releases/latest/download/trained_model.joblib
```
