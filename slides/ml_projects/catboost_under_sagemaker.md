# CatBoost on SageMaker — Full Pipeline (macOS)

> End-to-end guide: from trained model to live inference endpoint

---

## Overview

```
Trained .cbm model
       │
       ▼
  Package as model.tar.gz
       │
       ▼
  Upload to S3
       │
       ▼
  Configure AWS CLI + IAM Role
       │
       ▼
  Deploy SageMaker Endpoint
       │
       ▼
  Call via HTTP / boto3
```

---

## Part 1 — macOS Setup

### 1.1 Install AWS CLI

```bash
brew install awscli
aws --version   # aws-cli/2.x.x
```

### 1.2 Install Python dependencies

```bash
pip3 install sagemaker boto3 catboost
```

---

## Part 2 — IAM User + CLI Configuration

### 2.1 Create IAM user in AWS Console

```
https://console.aws.amazon.com/iam
IAM → Users → Create User
  Username: sagemaker-cli-user
  Policies:  AmazonSageMakerFullAccess
             AmazonS3FullAccess
```

### 2.2 Generate access key

```
IAM → Users → sagemaker-cli-user
  → Security credentials → Create access key → CLI use case
```

Download the `.csv` — you only see the Secret Key once.

### 2.3 Configure CLI

```bash
aws configure
# AWS Access Key ID:     AKIAIOSFODNN7EXAMPLE
# AWS Secret Access Key: wJalrXUtnFEMI/K7MDENG/...
# Default region name:   us-east-1
# Default output format: json
```

### 2.4 Verify

```bash
aws sts get-caller-identity
# Returns: UserId, Account, Arn
```

---

## Part 3 — SageMaker Execution Role

```bash
mkdir ~/sagemaker-setup && cd ~/sagemaker-setup
```

### 3.1 Create trust policy

```bash
cat > sagemaker-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "sagemaker.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF
```

### 3.2 Create the role

```bash
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document file://sagemaker-trust-policy.json
```

### 3.3 Attach policies

```bash
# SageMaker
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# S3 (full — swap for scoped policy in production)
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### 3.4 (Recommended) Scope S3 to your bucket

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME="your-bucket-name"

cat > s3-scoped-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::${BUCKET_NAME}",
      "arn:aws:s3:::${BUCKET_NAME}/*"
    ]
  }]
}
EOF

aws iam create-policy \
  --policy-name SageMakerS3ScopedPolicy \
  --policy-document file://s3-scoped-policy.json

aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/SageMakerS3ScopedPolicy
```

### 3.5 Save the Role ARN

```bash
ROLE_ARN=$(aws iam get-role \
  --role-name SageMakerExecutionRole \
  --query 'Role.Arn' \
  --output text)
echo $ROLE_ARN
# arn:aws:iam::123456789012:role/SageMakerExecutionRole
```

---

## Part 4 — Package & Upload Model

### 4.1 Directory structure

```
model/
├── model.cbm           ← CatBoost model (saved with model.save_model())
└── code/
    ├── inference.py
    └── requirements.txt
```

> ⚠️ `.cd` is a column description file, NOT the model. Save your model with:
> ```python
> model.save_model("model.cbm")
> ```

### 4.2 Inference script — `code/inference.py`

```python
import json, os
import numpy as np
from catboost import CatBoostClassifier  # or CatBoostRegressor

def model_fn(model_dir):
    model = CatBoostClassifier()
    model.load_model(os.path.join(model_dir, "model.cbm"))
    return model

def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        data = json.loads(request_body)
        features = data["features"]
        return features if isinstance(features[0], list) else [features]
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    return {
        "predictions": model.predict(input_data).tolist(),
        "probabilities": model.predict_proba(input_data).tolist()
    }

def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
```

### 4.3 `code/requirements.txt`

```
catboost==1.2.7
numpy
```

### 4.4 Package and upload

```bash
tar -czvf model.tar.gz model.cbm code/

aws s3 cp model.tar.gz s3://your-bucket-name/models/catboost/model.tar.gz
```

---

## Part 5 — Deploy SageMaker Endpoint

```python
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

model = SKLearnModel(
    model_data="s3://your-bucket-name/models/catboost/model.tar.gz",
    role=role,
    entry_point="inference.py",
    source_dir="code/",
    framework_version="1.2-1",
    py_version="py3",
)

# Free-tier eligible instance (125 hrs/month, first 2 months)
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name="catboost-inference-endpoint",
)
```

Deployment takes ~3–7 minutes.

### Serverless alternative (pay-per-call, no idle cost)

```python
from sagemaker.serverless import ServerlessInferenceConfig

predictor = model.deploy(
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=1024,
        max_concurrency=5,
    ),
    endpoint_name="catboost-serverless-endpoint",
)
```

---

## Part 6 — Call the Endpoint

### Python (boto3)

```python
import boto3, json

client = boto3.client("sagemaker-runtime", region_name="us-east-1")

response = client.invoke_endpoint(
    EndpointName="catboost-inference-endpoint",
    ContentType="application/json",
    Body=json.dumps({"features": [5.1, 3.5, 1.4, 0.2]})
)

result = json.loads(response["Body"].read())
print(result)
# {"predictions": [0], "probabilities": [[0.85, 0.15]]}
```

### AWS CLI

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name catboost-inference-endpoint \
  --content-type application/json \
  --body '{"features": [5.1, 3.5, 1.4, 0.2]}' \
  response.json && cat response.json
```

### Batch input

```json
{ "features": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]] }
```

---

## Part 7 — Cleanup

```bash
# Python
predictor.delete_endpoint()

# CLI
aws sagemaker delete-endpoint \
  --endpoint-name catboost-inference-endpoint
```

> ⚠️ SageMaker charges **per hour even when idle** — always delete when not in use.

---

## Quick Reference

| Component | Detail |
|---|---|
| Model format | `.cbm` (CatBoost native) |
| Artifact | `model.tar.gz` → S3 |
| Container | SageMaker SKLearn (`1.2-1`) |
| Instance | `ml.t2.medium` (free tier) |
| Free tier | 125 hrs/month, first 2 months |
| Input format | `{"features": [...]}` JSON |
| Output format | `{"predictions": [...], "probabilities": [...]}` |
| Idle cost | ~$0.065/hr after free tier |

---

## Checklist


✅ brew install awscli
✅ aws configure with Access Key + region
✅ aws sts get-caller-identity succeeds
✅ SageMakerExecutionRole created with trust policy
✅ SageMaker + S3 policies attached
✅ model.cbm saved (not .cd)
✅ model.tar.gz packaged with inference.py
✅ model.tar.gz uploaded to S3
✅ Endpoint deployed on ml.t2.medium
✅ Test invocation returns predictions
✅ Endpoint deleted after testing
