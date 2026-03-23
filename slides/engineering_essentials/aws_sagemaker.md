# Sagemaker

![alt text](img/sagemaker.png)


```shell
aws configure

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin [640168434252.dkr.ecr.us-east-2.amazonaws.com](http://640168434252.dkr.ecr.us-east-2.amazonaws.com/)
```

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