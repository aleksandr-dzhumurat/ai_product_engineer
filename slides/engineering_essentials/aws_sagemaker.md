# Sagemaker

![alt text](img/sagemaker.png)


```shell
aws configure

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin [640168434252.dkr.ecr.us-east-2.amazonaws.com](http://640168434252.dkr.ecr.us-east-2.amazonaws.com/)
```

# AWS CLI Setup

## 1. Create Access Keys in AWS Console

Go to: **IAM → Users → your user → Security credentials → Create access key**

Copy the **Access Key ID** and **Secret Access Key** (you only see the secret once).

---

## 2. Configure AWS CLI Locally

```bash
aws configure
```

It will prompt you:
```
AWS Access Key ID:     AKIA...
AWS Secret Access Key: xxxxxxxx
Default region name:   eu-central-1   # or your region
Default output format: json
```

**Don't have AWS CLI installed?**

```bash
# Ubuntu/Debian
sudo apt install awscli

# or latest version via pip
pip install awscli --break-system-packages
```

---

## 3. Test It

```bash
# List your buckets
aws s3 ls

# List contents of a specific bucket
aws s3 ls s3://your-bucket-name/
```

---

## 4. Copy Data

```bash
# Local → S3
aws s3 cp ./myfile.txt s3://your-bucket-name/path/

# S3 → Local
aws s3 cp s3://your-bucket-name/path/myfile.txt ./

# Sync entire folder
aws s3 sync ./local-folder s3://your-bucket-name/folder/
```

---

## 5. Create an S3 Bucket

```bash
aws s3 mb s3://your-bucket-name
# or with explicit region
aws s3 mb s3://your-bucket-name --region eu-central-1
```

Verify it was created:

```bash
aws s3 ls
```

---

## Using a `.env` File for Credentials

**Save keys in `.env`:**

```bash
# .env
AWS_ACCESS_KEY_ID=AKIAxxxxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AWS_DEFAULT_REGION=eu-central-1
```

**Load it safely:**

```bash
set -a; source .env; set +a
aws s3 ls
```

**Always add `.env` to `.gitignore`:**

```bash
echo ".env" >> .gitignore
```

**Load in Python/Node.js apps:**

```python
# Python - boto3 will auto-read the env vars
from dotenv import load_dotenv
load_dotenv()
```

```javascript
// Node.js - install dotenv first: npm install dotenv
require('dotenv').config()
// AWS SDK will auto-read AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
```

> **Note:** `export $(cat .env | xargs)` can mangle special characters (`+`, `/`, `=`). Use `set -a; source .env; set +a` instead.

---

## Using a Project-Local `.aws/` Directory

```bash
export AWS_CONFIG_FILE=$(pwd)/.aws/config
export AWS_SHARED_CREDENTIALS_FILE=$(pwd)/.aws/credentials

mkdir -p .aws
aws configure
```

Add `.aws/` to `.gitignore`:

```bash
echo ".aws/" >> .gitignore
```

---

# AWS EMR

AWS offers several EC2 and EMR instance types optimized for deep learning, primarily featuring NVIDIA GPUs. Below is a breakdown of P2, P3, G3, and P4d instances and their suitability for deep learning tasks.

---

## P2 Instances (Older GPU Option)

- GPUs: NVIDIA K80
- Compute Power: Up to 16 GPUs per instance
- Memory: Up to 732 GB RAM
- Networking: 25 Gbps
- Use Cases: Entry-level deep learning, model training with moderate-sized datasets

Good for: Small-scale deep learning training or inference workloads, but outdated compared to newer instances.

---

## P3 Instances (High-Performance Deep Learning)

- GPUs: NVIDIA V100 (Volta architecture)
- Compute Power: Up to 8 GPUs per instance
- Memory: Up to 488 GB RAM
- Networking: 100 Gbps (for P3dn)
- Use Cases: Deep learning training, high-performance computing (HPC), AI research

Good for: Large-scale deep learning training (TensorFlow, PyTorch, MXNet), and distributed training with Horovod.

P3dn.24xlarge (special version) supports NVIDIA NVLink, improving GPU-to-GPU communication speeds.

---

## G3 Instances (Graphics & Rendering, Not Ideal for DL)

- GPUs: NVIDIA M60
- Compute Power: 1-4 GPUs
- Memory: Up to 488 GB RAM
- Networking: 25 Gbps
- Use Cases: 3D rendering, video encoding, virtual desktops (not ideal for DL)

Not recommended for deep learning due to lower FP16 performance compared to P3/P4 instances.

---

## P4d Instances (Latest & Most Powerful)

- GPUs: NVIDIA A100 (Ampere architecture)
- Compute Power: 8 GPUs per instance
- Memory: 1.1 TB RAM, 320 GB GPU memory
- Networking: 400 Gbps
- Storage: Up to 8 TB NVMe SSD
- Use Cases: Large-scale deep learning (GPT, LLMs, GANs), AI supercomputing

Best choice for: Massive deep learning training (e.g., GPT-3/4, Stable Diffusion, large vision models).

P4d vs P3:

- 3x more training throughput (A100 is faster than V100).
- Better efficiency & NVLink support.
- Cheaper per TFLOP for high-end workloads.