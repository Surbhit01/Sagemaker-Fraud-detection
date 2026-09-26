#!/bin/bash
set -e

REGION="us-east-1"
REPO_NAME="sagemaker-paysim-ecr"

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create ECR repository if it doesn't already exist
aws ecr describe-repositories \
  --repository-names "$REPO_NAME" \
  --region "$REGION" \
  || aws ecr create-repository \
  --repository-name "$REPO_NAME" \
  --region "$REGION"

# Login Docker to Amazon ECR
aws ecr get-login-password \
  --region "$REGION" \
  | docker login \
  --username AWS \
  --password-stdin \
  "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# Build Docker image. If you are building on mac system, specify the platform. (If not specified, it will build for arm arch)
docker build --platform linux/amd64 -t "$REPO_NAME:latest" .

# Tag image for ECR
docker tag \
  "$REPO_NAME:latest" \
  "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest"

# Push image to ECR
docker push \
  "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest"

echo "Image URI: $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest"

