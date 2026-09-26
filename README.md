# Fraud Detection on AWS SageMaker

A hands-on, practical AWS SageMaker course. We build a fraud-detection model on
the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) synthetic transaction dataset.

The model itself is deliberately simple (a binary XGBoost classifier). The point of this
repo isn't fraud detection. It's using a fast, uncomplicated problem to learn
the SageMaker platform end to end. This includes local-to-cloud connectivity, S3 data flow, three
different ways to train a model (built in images, custom training scripts and bring your own containers), 
hyperparameter tuning, and two ways to deploy it (serverless and realtime inference)

## Repo structure

```
Notebooks/    The notebooks for the different steps in the sagemaker pipeline (EDA, training, hyperparameter tuning and model deployment)
```
## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) is a synthetic log of mobile-money
transactions with a binary `isFraud` label:

- **Severe class imbalance** — fraud is a tiny fraction of transactions, so accuracy is a
  useless metric here; every module uses AUC instead.
- **A known leakage artifact** — for fraudulent `TRANSFER`/`CASH_OUT` transactions, the
  balance columns don't follow the same accounting consistency as legitimate ones (e.g.
  `newbalanceOrig` often drops straight to `0` regardless of `amount`). This makes those
  columns act as a near-perfect proxy for the label rather than genuine signal. If a
  trained model reports `validation-auc` near `1.0`, this is why.

## Prerequisites

- An AWS account with billing enabled
- Python 3.13 and [`uv`](https://docs.astral.sh/uv/)
- AWS CLI, configured (`aws configure` or SSO)
- Docker, only needed for BYOC training method and SageMaker local-mode runs

## Quickstart

```bash
uv sync
```

## Course modules

| # | Module | Notebook | Covers |
|---|---|---|---|
| 1 | Connection | [0_Connection-&-Data-upload/1_connection.ipynb](0_Connection-&-Data-upload/0_Connection/1_connection.ipynb) | Verifying local AWS credentials before touching the SDK |
| 1 | Connection | [0_Connection-&-Data-upload/2_load_data_s3.ipynb](Notebooks/0_Connection-&-Data-upload/1_load_data_s3.ipynb) | Uploading data to S3 |
| 2 | EDA | [1_EDA/1_eda_fraud_detection.ipynb](Notebooks/1_EDA/1_eda_fraud_detection.ipynb) | Class imbalance, transaction-type breakdown, the leakage caveat |
| 3a | Training — Built-in algorithm | [2_Model-Training/1.Builtin_xgboost/Builtin_xgboost.ipynb](Notebooks/2_Model-Training/1.Builtin_xgboost/Builtin_xgboost.ipynb) | Managed XGBoost container, no training code to write |
| 3b | Training — Script mode | [2_Model-Training/2.Custom_training_script/Script_mode.ipynb](Notebooks/2_Model-Training/2.Custom_training_script/Script_mode.ipynb) | Custom `train.py`, SageMaker manages the container; local mode for fast debugging |
| 3c | Training — Bring-your-own-container | [2_Model-Training/3.BYOC/byoc.ipynb](Notebooks/2_Model-Training/3.BYOC/byoc.ipynb) | Full Docker image, pushed to ECR, SageMaker only orchestrates |
| 4 | Hyperparameter Tuning | [4_Hyperparameter-Tuning/1_hyperparameter_tuning.ipynb](Notebooks/4_Hyperparameter-Tuning/1_hyperparameter_tuning.ipynb) | `HyperparameterTuner`, Bayesian search, picking the best trial |
| 5a | Deployment — Real-time endpoint | [3_Deployment/1_realtime_endpoint.ipynb](Notebooks/3_Deployment/1_realtime_endpoint.ipynb) | Always-on HTTPS endpoint, `predictor.predict()` vs. raw `boto3` |
| 5b | Deployment — Serverless inference | [3_Deployment/2_serverless_inference.ipynb](Notebooks/3_Deployment/2_serverless_inference.ipynb) | Pay-per-request endpoint, cold starts, when it fits better than real-time |


**NOTE**: ONCE YOU ARE DONE WITH THE MODEL TRANING AND DEPLOYMENT AND IF THE RESOURCES ARE NO LONGER BEING USED, PLEASE DELETE THEM.

