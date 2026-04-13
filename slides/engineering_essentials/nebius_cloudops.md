# Nebius AI CloudOps Certification — Complete Preparation Guide

> **Exam format:** 1 hour · Multiple choice (1 correct + 3 distractors) · Remote, AI-proctored
> **Role:** Keep GPU clusters running day-to-day. Deploy from templates, manage access, monitor dashboards, restart failed nodes, escalate when needed.

---

## Domain Weights at a Glance

| Domain | Weight | Focus |
|---|---|---|
| 1. Security, compliance and billing | ~20% | IAM, encryption, billing models |
| 2. Setting up and operating GPU clusters | ~35% | Compute, InfiniBand, Kubernetes, storage |
| 3. Running training and inference workloads | ~20% | Slurm/Soperator, distributed jobs, inference apps |
| 4. Platform automation and maintenance | ~25% | IaC, maintenance, observability |

---

# DOMAIN 1 — Security, Compliance and Billing (~20%)

## 1.1 Resource Hierarchy and IAM Overview

→ [IAM Overview](https://docs.nebius.com/iam/overview) · [Managing Projects](https://docs.nebius.com/iam/manage-projects)

Nebius AI Cloud organizes resources in a two-level hierarchy:

**Tenant** → **Projects** → Resources

- A **tenant** is your top-level workspace. It contains projects, users, groups, billing info, quotas, and audit logs. You cannot delete a tenant.
- A **[project](https://docs.nebius.com/iam/manage-projects)** is an isolated workspace for resources (VMs, clusters, buckets). Each project belongs to exactly one region.
- **[Service accounts](https://docs.nebius.com/iam/service-accounts/manage)** belong to projects and can only act on resources within that project.

Resources are owned by services (e.g., [Compute](https://docs.nebius.com/compute) manages VMs/disks; [Object Storage](https://docs.nebius.com/object-storage) manages buckets). The CLI command group tells you which service owns a resource: `nebius compute disk ...` = Compute service.

### Default IAM Groups (least → most access)

→ [Groups and roles reference](https://docs.nebius.com/iam/authorization/roles)

| Group | Can view | Can access data | Can manage |
|---|---|---|---|
| `auditors` | Certain types only | No | No |
| `viewers` | Most types | Yes | No |
| `editors` | Most types | Yes | Most types |
| `admins` | All types | Yes | All types |

> Key rule: Being on the tenant user list does NOT grant resource access. Users must be added to a group.

### Account Types

- **User accounts** — federated users (humans), log in via SSO/browser
- **[Service accounts](https://docs.nebius.com/iam/service-accounts/manage)** — machine identities for CLI/API automation

### Key IAM CLI Commands

```bash
# Create a service account
nebius iam service-account create --name my-sa --format json

# Get editors group ID (needed to grant access)
nebius iam group get-by-name \
  --name editors \
  --parent-id <tenant_id> \
  --format json | jq -r '.metadata.id'

# Add service account to editors group
nebius iam group-membership create \
  --parent-id <group_id> \
  --member-id <sa_id>

# Create authorized key (upload public key for SA)
nebius iam auth-public-key create \
  --account-service-account-id $SA_ID \
  --data "$(cat public.pem)" \
  --format json
```

## 1.2 Authentication Methods

→ [How to authenticate in Nebius AI Cloud interfaces](https://docs.nebius.com/iam/log-in)

### Service Account Key Types

| Key Type | Use Case | Docs |
|---|---|---|
| **[Authorized keys](https://docs.nebius.com/iam/service-accounts/authorized-keys)** (public/private key pair) | Obtaining IAM tokens; used with CLI and API | [→](https://docs.nebius.com/iam/service-accounts/authorized-keys) |
| **[Static keys](https://docs.nebius.com/iam/authorization/static-keys)** | AWS-compatible services (Object Storage), long-term credentials (up to 3 years) | [→](https://docs.nebius.com/iam/authorization/static-keys) |
| **[Access tokens](https://docs.nebius.com/iam/authorization/access-tokens)** | Authenticate user accounts in Nebius AI Cloud interfaces | [→](https://docs.nebius.com/iam/authorization/access-tokens) |

### CLI Profile Authentication

→ [Setting up the CLI](https://docs.nebius.com/cli/configure) · [CLI Quickstart](https://docs.nebius.com/cli/quickstart)

Two flows when creating a profile:

**User account (federation):**
```bash
nebius profile create \
  --profile $PROFILE_NAME \
  --endpoint api.nebius.cloud \
  --federation-endpoint auth.nebius.com \
  --parent-id $PROJECT_ID
# → Opens browser for SSO login
```

**Service account:**
```bash
nebius profile create \
  --endpoint api.nebius.cloud \
  --service-account-id $SA_ID \
  --public-key-id $PUBLIC_KEY_ID \
  --private-key-file $PRIVATE_KEY_PATH \
  --profile $SA_PROFILE_NAME \
  --parent-id $PROJECT_ID
```

### Impersonation

Run a single command as a service account without switching profiles:
```bash
nebius compute instance list -I <service_account_id>
```

## 1.3 Federations and SSO

→ [SSO with Microsoft Entra ID (SAML)](https://docs.nebius.com/iam/federations/saml-sso) · [Keycloak SSO](https://docs.nebius.com/iam/federations/configure-sso-keycloak) · [JumpCloud SSO](https://docs.nebius.com/iam/federations/configure-sso-jumpcloud)

A **federation** allows users to log in via single sign-on (SSO). Supported providers include [Keycloak](https://docs.nebius.com/iam/federations/configure-sso-keycloak), [JumpCloud](https://docs.nebius.com/iam/federations/configure-sso-jumpcloud), and [Microsoft Entra ID (SAML SSO)](https://docs.nebius.com/iam/federations/saml-sso). Federations belong to the tenant level.

When creating a CLI profile with federation: endpoint is `auth.nebius.com`, auth type is `federation`.

## 1.4 Encryption and Secret Management

→ [Encryption in Nebius AI Cloud](https://docs.nebius.com/security/encryption) · [Key Management Service (KMS)](https://docs.nebius.com/kms)

### Encryption Overview

All Nebius AI Cloud storage is encrypted at rest using **AES-256**. Keys are managed by the **[Key Management Service (KMS)](https://docs.nebius.com/kms)**.

Encryption uses two key layers:
- **DEK (Data Encryption Key)** — encrypts actual data, unique per storage object
- **KEK (Key Encryption Key)** — encrypts the DEK, managed by KMS

| Storage Type | Encryption Default | Infrastructure | Service Level |
|---|---|---|---|
| [Network SSD disk](https://docs.nebius.com/compute/storage/types#disks) | **Always on** (cannot disable) | ✓ (KMS KEK) | ✓ (Compute DEK) |
| Network SSD NRD disk | Optional (enable manually) | — | ✓ when enabled |
| Network SSD IO M3 disk | Optional (enable manually) | — | ✓ when enabled |
| [Shared filesystem](https://docs.nebius.com/compute/storage/types#shared-filesystems) | **Always on** (cannot disable) | ✓ (KMS DEK) | — |
| [Object Storage bucket](https://docs.nebius.com/object-storage) | **Always on** | ✓ (KMS KEK) | ✓ (Object Storage DEK) |
| WEKA filesystem | Optional (enable at creation only) | — | ✓ (XTS-AES-256) |

> Note: Disk encryption can reduce write performance by up to 15%.
> Note: WEKA encryption cannot be enabled after filesystem creation.

### MysteryBox (Secrets Management)

→ [MysteryBox docs](https://docs.nebius.com/mysterybox) *(CLI: `nebius mysterybox`)*

[MysteryBox](https://docs.nebius.com/mysterybox) stores sensitive data (API keys, tokens, certificates) as **secrets** in encrypted form. Secrets have versions; each version has payloads. This avoids hardcoding sensitive data in scripts and pipelines.

## 1.5 Billing Models

→ [Billing Models Overview](https://docs.nebius.com/signup-billing/billing-models/overview) · [PAYG](https://docs.nebius.com/signup-billing/billing-models/payg) · [Commitment discounts](https://docs.nebius.com/signup-billing/billing-models/committed-usage)

Nebius AI Cloud has two billing models:

**[Pay-as-you-go (PAYG)](https://docs.nebius.com/signup-billing/billing-models/payg)**
- Flexible, per-second billing for variable workloads
- Available to individuals and companies
- Default billing model
- Charges begin when a VM reaches `Running` status; charges stop when deletion command is sent

**[Commitment Discounts](https://docs.nebius.com/signup-billing/billing-models/committed-usage)**
- Discounts for committing to specific usage
- Available to companies only
- Fits stable, long-term workloads
- Can be combined with PAYG (PAYG covers anything not in commitment)

### Billing Threshold

→ [How a billing threshold works](https://docs.nebius.com/signup-billing/payments/threshold)

A spending limit that triggers automatic payment attempt on the card on file. You can configure [billing thresholds](https://docs.nebius.com/signup-billing/payments/threshold) to control cloud spend.

### What Stops/Starts Billing

- **Quotas** are only released when a VM is **deleted** (not stopped). A stopped VM still occupies quota.
- **Billing** stops when the delete command is sent (not when deletion completes).
- **Billing** is paused during VM crash recovery; resumes if recovery succeeds.

### Shared Responsibility

- **Nebius** is responsible for: physical infrastructure security, hypervisor, network fabric, encryption-at-rest, KMS
- **Customer** is responsible for: OS patching, application security, access management, data classification, network security group rules

---

# DOMAIN 2 — Setting Up and Operating GPU Clusters (~35%)

## 2.1 VM Types and GPU Platforms

→ [Types of virtual machines and GPUs](https://docs.nebius.com/compute/virtual-machines/types) · [List available platforms](https://docs.nebius.com/compute/virtual-machines/list-platforms) · [Capacity advisor](https://docs.nebius.com/compute/virtual-machines/capacity-advisor)

Nebius AI Cloud offers several GPU platforms. The **preset name** encodes resources: `8gpu-128vcpu-1600gb` = 8 GPUs, 128 vCPUs, 1600 GiB RAM.

| Platform ID | GPU | Architecture | InfiniBand | HBM Memory |
|---|---|---|---|---|
| `gpu-b300-sxm` | [NVIDIA B300](https://www.nvidia.com/en-us/data-center/dgx-platform/) | Blackwell Ultra | 800 Gbps (ConnectX-8) | 288 GB HBM3e |
| `gpu-b200-sxm` / `gpu-b200-sxm-a` | [NVIDIA B200](https://www.nvidia.com/en-us/data-center/hgx/) | Blackwell | 400 Gbps (ConnectX-7) | 180 GB HBM3e |
| `gpu-h200-sxm` | [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) | Hopper | 400 Gbps (ConnectX-7) | 141 GB HBM3e |
| `gpu-h100-sxm` | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) | Hopper | 400 Gbps (ConnectX-7) | 80 GB HBM3 |
| `gpu-rtx6000` | [NVIDIA RTX PRO 6000](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) | Blackwell PCIe | BlueField-3 400 Gbps | 96 GB GDDR7 |
| `gpu-l40s-a` / `gpu-l40s-d` | [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/) | Ada Lovelace PCIe | No InfiniBand | 48 GB GDDR6 |

**GPU cluster-compatible presets** (8-GPU nodes only for HPC):
- `gpu-h100-sxm`: `8gpu-128vcpu-1600gb`
- `gpu-h200-sxm`: `8gpu-128vcpu-1600gb`
- `gpu-b200-sxm`: `8gpu-160vcpu-1792gb`
- `gpu-b300-sxm`: `8gpu-192vcpu-2768gb`

Non-GPU platforms: `cpu-d3` (AMD EPYC Genoa) and `cpu-e2` (Intel Ice Lake).

Check available platforms in your project:
```bash
nebius compute platform list
```

## 2.2 InfiniBand and GPU Clusters

→ [InfiniBand networking for Compute VMs with GPUs](https://docs.nebius.com/compute/clusters/gpu) · [InfiniBand topology](https://docs.nebius.com/compute/clusters/gpu/topology) · [Manage topology](https://docs.nebius.com/compute/clusters/gpu/topology/manage)

### What is a GPU Cluster?

A **[GPU cluster](https://docs.nebius.com/compute/clusters/gpu)** groups VMs into a high-speed InfiniBand fabric. It accelerates HPC tasks like distributed training that require more processing power than a single VM can provide.

### InfiniBand Architecture

- Each GPU in a VM is connected via a NIC providing **400 Gbps** (H100/H200/B200) or **800 Gbps** (B300)
- An 8-GPU node total bandwidth = **3.2 Tbps** (H/B200) or higher (B300)
- Uses **GPUDirect RDMA** — data flows directly between GPU and NIC, bypassing the CPU
- **Partition Keys (P-Keys)** isolate InfiniBand traffic between different GPU clusters, even on the same physical fabric

### InfiniBand Fabrics

Each GPU cluster is assigned to a physical [InfiniBand fabric](https://docs.nebius.com/compute/clusters/gpu/topology). Select the fabric matching your GPU type and region:

| Fabric | GPU Platform | Region |
|---|---|---|
| `fabric-2/3/4/6` | H100 SXM | `eu-north1` |
| `fabric-5` | H200 SXM | `eu-west1` |
| `fabric-7` | H200 SXM | `eu-north1` |
| `eu-north2-a` | H200 SXM | `eu-north2` |
| `me-west1-a` | B200 SXM | `me-west1` |
| `us-central1-a` | H200 SXM | `us-central1` |
| `us-central1-b` | B200 SXM | `us-central1` |
| `uk-south1-a` | B300 SXM | `uk-south1` |

> In most cases, use the preselected fabric. Change only if you need a specific platform or face capacity issues.

### Creating a GPU Cluster (CLI)

→ [How to create a VM](https://docs.nebius.com/compute/virtual-machines/manage) · [Boot disk images](https://docs.nebius.com/compute/storage/boot-disk-images)

```bash
# 1. Set project and fabric
nebius profile update --parent-id <project_id>
export INFINIBAND_FABRIC=fabric-7

# 2. Create GPU cluster
export GPU_CLUSTER_ID=$(nebius compute gpu-cluster create \
  --name my-gpu-cluster \
  --infiniband-fabric $INFINIBAND_FABRIC \
  --format json | jq -r ".metadata.id")

# 3. Create boot disk (use GPU-compatible image)
export BOOT_DISK_ID=$(nebius compute disk create \
  --name my-boot-disk \
  --size-gibibytes 200 \
  --type network_ssd \
  --source-image-family-image-family ubuntu24.04-cuda13.0 \
  --block-size-bytes 4096 \
  --format json | jq -r ".metadata.id")

# 4. Create VM and join it to the cluster
nebius compute instance create \
  --resources-platform gpu-h200-sxm \
  --resources-preset 8gpu-128vcpu-1600gb \
  --gpu-cluster-id $GPU_CLUSTER_ID \
  --boot-disk-existing-disk-id $BOOT_DISK_ID \
  ...

# 5. List and delete clusters
nebius compute gpu-cluster list
nebius compute gpu-cluster delete <GPU_cluster_ID>
```

> All VMs in a GPU cluster (including Kubernetes nodes) must belong to the **same project**.

### Testing InfiniBand with NCCL

Run NCCL all-reduce tests to validate InfiniBand fabric performance:
- In Compute VMs: [MPIrun tutorial with NCCL test](https://docs.nebius.com/3p-integrations/mpirun)
- In Kubernetes: [Running NCCL tests in Kubernetes](https://docs.nebius.com/kubernetes/gpu/nccl-test)
- In Soperator: [NCCL all-reduce in Soperator](https://docs.nebius.com/slurm-soperator/jobs/examples/nccl-all-reduce)

## 2.3 VM Lifecycle

→ [Lifecycle and statuses of VMs](https://docs.nebius.com/compute/virtual-machines/lifecycle) · [Stopping and starting VMs](https://docs.nebius.com/compute/virtual-machines/stop-start) · [Maintenance overview](https://docs.nebius.com/compute/virtual-machines/maintenance)

VM statuses and transitions:

```
Stopped → Starting → Running → Stopping → Stopped
                              → Deleting → (removed)
                    → Error   (delete and recreate)
```

Key facts:
- Create a VM in `Stopped` state: add `"stopped": true` in CLI/Terraform config
- Graceful stop: hypervisor waits up to **60 seconds** for processes to terminate
- `Error` status: VM cannot be recovered; delete it and create a new one
- **Quotas** are occupied until deletion — stopping does NOT release quotas
- **Billing** starts at `Running`, stops when the delete command is sent

## 2.4 Storage — Disks and Filesystems

→ [Types of storage volumes in Compute](https://docs.nebius.com/compute/storage/types) · [Managing volumes](https://docs.nebius.com/compute/storage/manage) · [Attaching volumes to VMs](https://docs.nebius.com/compute/storage/use)

### Disk Types

| Type | CLI ID | Performance | Reliability | Best For |
|---|---|---|---|---|
| [Network SSD](https://docs.nebius.com/compute/storage/types#disks) | `network_ssd` | 450 MiB/s, 40K IOPS write | Erasure coding (2 failures) | Boot disks, controller VMs |
| Network SSD NRD | `network_ssd_non_replicated` | 1 GiB/s, 75K IOPS | None | K8s node disks, temp storage |
| Network SSD IO M3 | `network_ssd_io_m3` | 1 GiB/s, 75K IOPS | Mirrored ×3 | GlusterFS, DB hosts |

Performance comparison: **SSD IO M3 = SSD NRD > SSD**
Reliability: **SSD IO M3 > SSD > SSD NRD**
Price: **SSD NRD < SSD < SSD IO M3**

> SSD NRD disk size must be a multiple of **93 GiB**.
> SSD encryption is **always on**. SSD NRD and IO M3 encryption is optional.

### Shared Filesystems

→ [Shared filesystems](https://docs.nebius.com/compute/storage/types#shared-filesystems)

- One [shared filesystem](https://docs.nebius.com/compute/storage/types#shared-filesystems) can be attached to **multiple VMs simultaneously** (must be in the same project)
- Mount as **virtiofs** device on the VM
- Encryption is **always on** (cannot be disabled)
- Capacity: up to **5 PiB**
- Max read bandwidth per client: **12 GiB/s** | write: **8 GiB/s**
- Aggregate read: **940 GiB/s** | aggregate write: **480 GiB/s**
- Performance increases every 4 TiB of filesystem size

### Storage Selection by Use Case

| Use Case | Recommended Storage |
|---|---|
| VM boot disk | [Network SSD](https://docs.nebius.com/compute/storage/types#disks) |
| Kubernetes node disk | Network SSD NRD |
| Database host | Network SSD IO M3 |
| GlusterFS storage disk | Network SSD IO M3 |
| Dataset storage / preprocessing | [Object Storage](https://docs.nebius.com/object-storage) |
| Streaming datasets to training workers | [SSD shared filesystem](https://docs.nebius.com/compute/storage/types#shared-filesystems) |
| Sharing code between workers | SSD shared filesystem |
| Checkpointing during training | SSD shared filesystem → then Object Storage |
| Sharing inference weights across GPU nodes | SSD shared filesystem |
| Sharing inference results | [Object Storage](https://docs.nebius.com/object-storage) |

### Object Storage

→ [Object Storage overview](https://docs.nebius.com/object-storage)

[Object Storage](https://docs.nebius.com/object-storage) is a separate service for unstructured data (datasets, model weights, outputs). Access via `nebius storage bucket` commands. Use [static keys](https://docs.nebius.com/iam/authorization/static-keys) (S3-compatible API) or access tokens.

## 2.5 Network Connectivity (VPC)

→ [VPC overview](https://docs.nebius.com/vpc) · [Custom NAT gateway](https://docs.nebius.com/vpc/routing/custom-nat-gateway) · [Security groups](https://docs.nebius.com/terraform-provider/reference/resources/vpc_v1_security_group)

VPC resources managed under `nebius vpc`:

- **Network** — virtual private network
- **Subnet** — IP range within a network and region
- **Allocation** — reserved IP address
- **Pool** — address pool
- **Security group** — stateful firewall rules
- **Routing table / Route** — custom routing

```bash
nebius vpc subnet list
nebius vpc network list
```

[NAT gateways](https://docs.nebius.com/vpc/routing/custom-nat-gateway) can be configured for outbound internet access from private subnets.

## 2.6 Managed Kubernetes (mk8s)

→ [Managed Kubernetes overview](https://docs.nebius.com/kubernetes) · [Quickstart](https://docs.nebius.com/kubernetes/quickstart) · [Cluster components](https://docs.nebius.com/kubernetes/components)

### Cluster Components

A Kubernetes cluster contains:
- **Control plane** — managed by Nebius (API server, scheduler, etcd)
- **[Node groups](https://docs.nebius.com/kubernetes/node-groups/manage)** — groups of Compute VMs serving as worker nodes

### Creating GPU Node Groups

→ [Working with GPUs in Managed Kubernetes](https://docs.nebius.com/kubernetes/gpu/set-up) · [Creating node groups](https://docs.nebius.com/kubernetes/node-groups/manage)

```bash
# Create node group with GPU drivers
nebius mk8s node-group create \
  --template-resources-platform gpu-h200-sxm \
  --template-resources-preset 8gpu-128vcpu-1600gb \
  --template-gpu-settings-drivers-preset cuda12.8 \
  ...

# Check compatibility matrix for your K8s version and platform
nebius mk8s node-group get-compatibility-matrix \
  --cluster-kubernetes-version 1.33 \
  --platform gpu-h200-sxm

# Get kubectl credentials
nebius mk8s cluster get-credentials \
  --id <cluster_id> --external

# Update driver preset (triggers node recreation)
nebius mk8s node-group update \
  --id <group_id> \
  --template-gpu-settings-drivers-preset cuda13.0
```

### Driver Presets

| Preset | NVIDIA Driver | OS |
|---|---|---|
| `cuda12.8` | 570.x | `ubuntu24.04` |
| `cuda13.0` | 580.x | `ubuntu24.04` |

> Kubernetes 1.30 (deprecated): use `cuda12`. K8s 1.31+ use `cuda12.8`.

### Manual GPU Operator Installation

→ [GPU operator installation guide](https://docs.nebius.com/kubernetes/gpu/set-up#how-to-install-the-drivers-and-components-on-existing-node-groups)

If not using the built-in driver image, install via Helm from the Nebius Marketplace chart repository (`cr.eu-north1.nebius.cloud`):

**With InfiniBand:**
1. Install NVIDIA **Network Operator** first
2. Verify `NICClusterPolicy` reaches `"state": "ready"` and `state-OFED` is `ready`
3. Install NVIDIA **GPU Operator**
4. Verify driver DaemonSets log `Done, now waiting for signal`

**Without InfiniBand:**
- Install only NVIDIA **GPU Operator**

> Install and verify operators in order — they depend on each other.

### Autoscaling

→ [Autoscaling in Managed Kubernetes](https://docs.nebius.com/kubernetes/node-groups/autoscaling)

[Managed Kubernetes supports node group autoscaling](https://docs.nebius.com/kubernetes/node-groups/autoscaling) — the cluster scales out during compute-heavy phases and scales down during idle periods.

### Topology-Aware Scheduling

→ [Topology-aware scheduling for GPU workloads](https://docs.nebius.com/kubernetes/gpu/topology-aware-scheduling)

For GPU workloads with InfiniBand, Kubernetes supports [topology-aware scheduling](https://docs.nebius.com/kubernetes/gpu/topology-aware-scheduling) to place pods on nodes within the same InfiniBand switch domain, minimizing inter-node latency.

---

# DOMAIN 3 — Running Training and Inference Workloads (~20%)

## 3.1 Orchestration Model Selection

→ [Why Slurm and Soperator?](https://docs.nebius.com/slurm-soperator/overview/why-slurm-soperator) · [Running AI workloads on VMs](https://docs.nebius.com/compute/clusters/ai-workloads)

Nebius AI Cloud offers two main orchestration models for ML workloads:

| Aspect | [Slurm / Soperator](https://docs.nebius.com/slurm-soperator) | [Managed Kubernetes](https://docs.nebius.com/kubernetes) |
|---|---|---|
| Model | Traditional HPC job scheduler | Container orchestrator |
| Best for | Multi-node distributed training, batch ML jobs | Inference serving, microservices, mixed workloads |
| Job submission | `sbatch` | `kubectl apply`, Helm, Kubeflow |
| Shared filesystem | Shared root filesystem (all nodes see same `/`) | PersistentVolumes, shared filesystems |
| Scaling | Auto-scales Kubernetes workers backing Slurm nodes | Node group autoscaling |
| Containers | Supported via `--container-image` in `srun` | Native |

## 3.2 Slurm and Soperator Architecture

→ [Soperator architecture](https://docs.nebius.com/slurm-soperator/overview/architecture) · [Slurm and Soperator overview](https://docs.nebius.com/slurm-soperator)

**[Soperator](https://docs.nebius.com/slurm-soperator/overview/architecture)** deploys Slurm onto a Kubernetes cluster. Slurm nodes are Kubernetes Pods.

### Node Types

| Node Type | Daemon | Role |
|---|---|---|
| **Login nodes** | `sshd` | User entry point — submit jobs, check status, prepare data |
| **Worker nodes** (compute nodes) | `slurmd` | Execute Slurm job steps |
| **Controller nodes** | `slurmctld` | Job queuing, resource allocation, node monitoring |

Load is balanced across login nodes — each SSH connection goes to a random login node.

### Shared Root Filesystem

Soperator's key feature: all login and worker nodes share a **single root filesystem** (mounted as `/`). Changes on one node instantly appear on all others. No manual synchronization needed.

Implemented as a Kubernetes PersistentVolume (PV).

### Deployment Methods

→ [Deployment methods overview](https://docs.nebius.com/slurm-soperator/deploy/overview) · [Managed Service for Soperator](https://docs.nebius.com/slurm-soperator/managed-soperator/manage)

1. **[Managed Service for Soperator](https://docs.nebius.com/slurm-soperator/managed-soperator/manage)** — Nebius-managed, easiest to operate
2. **Pro Solution** — deployed by Nebius solution architects
3. **Self-deployed** on Managed Kubernetes
4. Other cloud or on-premises

## 3.3 Running Slurm Jobs

→ [Running Slurm batch jobs](https://docs.nebius.com/slurm-soperator/jobs) · [Connecting to login and worker nodes](https://docs.nebius.com/slurm-soperator/clusters/connect)

### Job Workflow

```bash
# 1. Connect to a login node
ssh <login_node_address>

# 2. Submit a batch job
sbatch --nodes=4 my_ml_job.sh
# Output: Submitted batch job 610

# 3. Monitor output
tail -f output_610.txt

# 4. Check job status
squeue  # or scontrol show job <job_id>
```

### Batch Script Structure

```bash
#!/bin/bash

# Job configuration directives
#SBATCH --job-name=llm_training
#SBATCH --output=%x_%j.out        # %x=job name, %j=job ID
#SBATCH --error=%x_%j.err
#SBATCH --nodes=2                  # number of worker nodes
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00            # time limit (HH:MM:SS)
#SBATCH --exclusive                # all CPUs on nodes → no other jobs share
#SBATCH --mem=0                    # all available RAM

# Multi-node step — runs command in parallel on all nodes
srun --cpus-per-task=16 python train.py
```

### Key `sbatch` Settings Reference

| Parameter | Short | Meaning |
|---|---|---|
| `--job-name` | `-J` | Job name |
| `--nodes` | `-N` | Number of worker nodes |
| `--nodelist` | `-w` | Specific nodes to use |
| `--exclude` | `-x` | Nodes to exclude |
| `--output` | `-o` | Stdout file (supports `%j`, `%x` patterns) |
| `--error` | `-e` | Stderr file |
| `--time` | `-t` | Time limit (`01:00` = 1hr, `1-00` = 1 day) |
| `--gpus-per-node` | | GPUs per node |
| `--ntasks-per-node` | | Max tasks per node |
| `--exclusive` | | Allocate all CPUs; prevents job sharing |
| `--cpus-per-task` | `-c` | CPUs per task |
| `--mem` | `-m` | RAM per node (`0` = all available) |
| `--partition` | `-p` | Slurm partition |
| `--requeue` | | Auto-requeue on node failure |
| `--dependency` | `-d` | Job dependencies (`afterany:20:21`, `singleton`) |
| `--parsable` | | Output just job ID |

### Configuration Priority (highest → lowest)

1. Command parameters (`sbatch --time=02:00:00`)
2. Session environment variables (`export SBATCH_TIMELIMIT=...`)
3. User profile (`~/profile`)
4. Cluster profile (`/etc/profile`)
5. `#SBATCH` directives in script
6. `~/.slurm/defaults`

### Key Output Environment Variables in Jobs

| Variable | Value |
|---|---|
| `SLURM_JOB_NODELIST` | List of allocated worker nodes |
| `SLURM_NNODES` | Number of worker nodes |
| `SLURM_NODEID` | ID of current node (per-node) |
| `SLURM_SUBMIT_DIR` | Directory where `sbatch` was run |
| `SLURMDAEMON_NODENAME` | Hostname of node running `sbatch` script |

### Distributed Training Examples

**Hugging Face Accelerate** (2 nodes × 8 GPUs):
```bash
srun --cpus-per-task=64 --hint=nomultithread \
  bash -c 'accelerate launch \
    --num_machines $SLURM_STEP_NUM_NODES \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $MAIN_PROCESS_ADDR \
    --num_processes $((SLURM_STEP_NUM_NODES * SLURM_GPUS_ON_NODE)) \
    train.py'
```

**PyTorch torchrun** (3 nodes × 8 GPUs):
```bash
torchrun --nnodes $SLURM_NNODES \
  --nproc_per_node 8 \
  --master_addr $HOST_ADDR \
  --node_rank=$SLURM_NODEID \
  finetuning.py ...
```

**Containerized jobs** (→ [Running jobs in containers](https://docs.nebius.com/slurm-soperator/jobs/containers/index)):
```bash
srun --container-image="nvcr.io#nvidia/tensorflow:23.02-tf1-py3" \
  python -c "import tensorflow as tf; print(tf.__version__)"
```

## 3.4 Connecting Jobs to Persistent and Shared Storage

→ [Downloading data in Soperator clusters](https://docs.nebius.com/slurm-soperator/storage/download-data) · [Managing file access](https://docs.nebius.com/slurm-soperator/storage/manage-access)

In Soperator clusters, all nodes share the root filesystem — data is automatically available on all nodes without explicit copying.

For large datasets or external data:
- [Download data to the login node](https://docs.nebius.com/slurm-soperator/storage/download-data) (appears on all nodes automatically via shared FS)
- Mount [Object Storage](https://docs.nebius.com/object-storage) buckets via S3-compatible tools
- Use [shared filesystems](https://docs.nebius.com/compute/storage/types#shared-filesystems) mounted as PVs

Storage access from Kubernetes jobs:
- PersistentVolumeClaims (PVCs) backed by Nebius shared filesystems
- Object Storage accessed via S3 API with [static keys](https://docs.nebius.com/iam/authorization/static-keys)

## 3.5 Inference Apps (Serverless AI)

→ [Serverless AI overview](https://docs.nebius.com/serverless/overview) · [Managing endpoints](https://docs.nebius.com/serverless/endpoints/manage)

Nebius AI Cloud offers **[Serverless AI](https://docs.nebius.com/serverless/overview)** with ready-made inference endpoints. You can:
- Deploy inference endpoints without managing underlying infrastructure
- Scale endpoints automatically
- Connect to shared filesystems and Object Storage for model weights

```bash
nebius serverless endpoint list
nebius serverless endpoint create ...
```

---

# DOMAIN 4 — Platform Automation and Maintenance (~25%)

## 4.1 Infrastructure as Code (Terraform)

→ [Terraform provider overview](https://docs.nebius.com/terraform-provider) · [Quickstart](https://docs.nebius.com/terraform-provider/quickstart) · [Authentication](https://docs.nebius.com/terraform-provider/authentication)

### Provider Setup

**`terraform.tf`** — declare the provider:
```hcl
terraform {
  required_providers {
    nebius = {
      source  = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
      version = ">= 0.5.55"
    }
  }
}
```

**`providers.tf`** — authenticate as a service account:
```hcl
provider "nebius" {
  service_account = {
    private_key_file_env = "AUTHKEY_PRIVATE_PATH"
    public_key_id_env    = "AUTHKEY_PUBLIC_ID"
    account_id_env       = "SA_ID"
  }
}
```

Alternatively, [authenticate with a user account](https://docs.nebius.com/terraform-provider/authentication#authenticating-with-your-user-account).

### Terraform Workflow

```bash
terraform init      # download provider plugin
terraform validate  # check config syntax
terraform plan      # preview changes (dry run)
terraform apply     # apply changes
terraform destroy   # tear down all resources
```

### Key Terraform Resources

| Resource | What It Creates | Docs |
|---|---|---|
| `nebius_compute_v1_instance` | Virtual machine | [→](https://docs.nebius.com/terraform-provider/reference/resources/compute_v1_instance) |
| `nebius_compute_v1_disk` | Block disk | [→](https://docs.nebius.com/terraform-provider/reference/resources/compute_v1_disk) |
| `nebius_mk8s_v1_node_group` | Kubernetes node group | [→](https://docs.nebius.com/terraform-provider/reference/resources/mk8s_v1_node_group) |
| `nebius_iam_v1_service_account` | IAM service account | [→](https://docs.nebius.com/terraform-provider/reference/resources/iam_v1_service_account) |
| `nebius_vpc_v1_pool` | VPC subnet/address pool | [→](https://docs.nebius.com/terraform-provider/reference/resources/vpc_v1_pool) |
| `nebius_vpc_v1_security_group` | VPC security group | [→](https://docs.nebius.com/terraform-provider/reference/resources/vpc_v1_security_group) |
| `nebius_vpc_v1_allocation` | IP address allocation | [→](https://docs.nebius.com/terraform-provider/reference/data-sources/vpc_v1_allocation) |
| `nebius_storage_v1_bucket` | Object Storage bucket | [→](https://docs.nebius.com/terraform-provider/reference/resources/storage_v1_bucket) |
| `nebius_registry_v1_registry` | Container registry | [→](https://docs.nebius.com/terraform-provider) |

Every resource requires `parent_id` = project ID.

### Storing Terraform State in Object Storage

→ [Storing Terraform state in Object Storage](https://docs.nebius.com/terraform-provider/store-terraform-state)

Remote state can be stored in a [Nebius Object Storage bucket](https://docs.nebius.com/terraform-provider/store-terraform-state). This allows collaboration across teams and prevents state file loss.

### Pulumi Support

→ [Using the provider with Pulumi](https://docs.nebius.com/terraform-provider/pulumi)

Nebius also supports [Pulumi via the Terraform provider bridge](https://docs.nebius.com/terraform-provider/pulumi).

### Sensitive Values

→ [Working with sensitive values](https://docs.nebius.com/terraform-provider/sensitive-values)

Use [sensitive value patterns](https://docs.nebius.com/terraform-provider/sensitive-values) to avoid leaking secrets into the Terraform state file.

## 4.2 VM Maintenance

→ [VM maintenance overview](https://docs.nebius.com/compute/virtual-machines/maintenance) · [Maintenance reason codes](https://docs.nebius.com/compute/virtual-machines/maintenance-reasons) · [Preemptible VMs](https://docs.nebius.com/compute/virtual-machines/preemptible)

### Maintenance Events

VMs undergo planned maintenance (hardware updates, migrations). Compute exposes:
- **[Maintenance reason codes](https://docs.nebius.com/compute/virtual-machines/maintenance-reasons)** explaining why an event was triggered
- Ability to view and respond to scheduled maintenance

```bash
nebius maintenance list              # list maintenance events
```

[Preemptible VMs](https://docs.nebius.com/compute/virtual-machines/preemptible) have a specific behavior:
- `on_preemption` parameter: only supported value is `STOP` (stops VM, does not delete/restart)

### Stopping and Starting VMs

→ [Stopping and starting VMs](https://docs.nebius.com/compute/virtual-machines/stop-start)

```bash
nebius compute instance stop <id>
nebius compute instance start <id>
nebius compute instance restart <id>
```

### Automatic Security Updates

→ [Enabling automatic security updates](https://docs.nebius.com/compute/storage/automatic-updates)

Can be enabled per VM for OS-level patches.

### Managed Kubernetes Maintenance

→ [Enable/disable health checks](https://docs.nebius.com/kubernetes/maintenance/enable-disable-health-checks)

K8s cluster maintenance includes:
- [Health checks](https://docs.nebius.com/kubernetes/maintenance/enable-disable-health-checks) enabled by default for clusters created from Dec 1, 2025
- Health checks can be enabled/disabled: `nebius mk8s cluster update --enable-health-checks`
- Node group updates trigger rolling node recreation (cordon → drain → delete → replace)

## 4.3 Observability Stack

→ [Observability overview](https://docs.nebius.com/observability) · [Alerts](https://docs.nebius.com/observability/alerts) · [Metrics](https://docs.nebius.com/observability/monitoring) · [Logs](https://docs.nebius.com/observability/logging)

[Nebius AI Cloud Observability](https://docs.nebius.com/observability) is a **unified solution** combining alerts, metrics, and logs in a single entry point — the **Observability Overview page** in the web console.

### Three Core Components

**1. [Firing Alerts](https://docs.nebius.com/observability/alerts)** — active alerts in your project (surfaced immediately on the overview page)

**2. [Metrics](https://docs.nebius.com/observability/monitoring)** — resource performance indicators (CPU, memory, disk, network, GPU utilization):
- Preconfigured dashboards in the web console
- **[Grafana](https://docs.nebius.com/observability/metrics/grafana)** integration for advanced visualization
- **[Prometheus](https://docs.nebius.com/observability/metrics/prometheus)** integration for metric scraping and alerting
- [Monitoring agent](https://docs.nebius.com/observability/agents/monitoring-agent) (`nebius-o11y-agent`) collects from Compute VMs and Kubernetes

**3. [Logs](https://docs.nebius.com/observability/logging)** — recent logs from services for debugging:
- [Query language](https://docs.nebius.com/observability/logs/query-language) for filtering
- Export logs or ingest custom logs
- [Kubernetes logs](https://docs.nebius.com/kubernetes/logs)

Additional: **[Tracing](https://docs.nebius.com/observability/traces)** (for Kubernetes applications)

### Monitoring Agent

→ [Monitoring agent on Compute VMs](https://docs.nebius.com/observability/agents/monitoring-agent) · [Nebius Observability Agent for Kubernetes](https://docs.nebius.com/observability/metrics/ingest/nebius-o11y-agent)

Nebius provides a **[monitoring agent](https://docs.nebius.com/observability/agents/monitoring-agent)** with each Compute VM. It collects:
- GPU utilization
- vCPU utilization
- RAM usage

Data is visualized on dashboards in the web console. For Kubernetes, use the **[Nebius Observability Agent for Kubernetes](https://docs.nebius.com/observability/metrics/ingest/nebius-o11y-agent)** (`nebius-o11y-agent`) to ingest cluster metrics.

### GPU Health Checks

- GPU health checks run **before and after each Slurm job**
- In Managed Kubernetes, health checks are **enabled by default** for clusters created December 1, 2025 or later ([manage health checks](https://docs.nebius.com/kubernetes/maintenance/enable-disable-health-checks))
- Check node health via kubectl: `kubectl get nodes` — look for `Ready` / `NotReady` status
- Check GPU operator: DaemonSet logs must show `Done, now waiting for signal`

## 4.4 Diagnosing Network and Storage Performance Issues

→ [Observability agents](https://docs.nebius.com/observability/agents) · [NCCL tests in Kubernetes](https://docs.nebius.com/kubernetes/gpu/nccl-test)

### InfiniBand / Network Diagnostics

- Run **[NCCL all-reduce tests](https://docs.nebius.com/kubernetes/gpu/nccl-test)** to benchmark InfiniBand bandwidth and latency
- Check `NICClusterPolicy` status for Network Operator health: `kubectl get nicclusterpolicy.mellanox.com nic-cluster-policy -o json | jq -r '.status'`
- MOFED (Mellanox OFED) driver logs during installation: `kubectl logs -n nvidia-network-operator $(kubectl get pods -n nvidia-network-operator | grep mofed | head -1 | awk '{print $1}')`

### Storage Performance

- Match I/O block size to the volume's **block size** for maximum IOPS
- Use **4 MiB chunk sizes** for maximum bandwidth
- SSD NRD/IO M3 performance scales with disk size (per 93 GiB allocation unit)
- SSD performance scales with size (per 32 GiB allocation unit)
- Encryption reduces write performance by up to **15%** ([Encryption docs](https://docs.nebius.com/security/encryption))

### Observability CLI

→ [CLI reference: logging](https://docs.nebius.com/cli/reference/logging)

```bash
nebius logging log-group list
nebius logging entry list --log-group-id <id>
```

---

# Practice Questions

## Domain 1 — Security, Compliance and Billing

**Q1.** A new team member is added to the tenant user list but cannot access any resources. What is the most likely cause?
> They have not been added to any IAM group. Being on the tenant user list does not grant resource access. → [IAM Overview](https://docs.nebius.com/iam/overview)

**Q2.** What is the difference between `editors` and `admins` groups?
> `editors` can view and manage most resource types. `admins` can view and manage ALL resource types, including access management and security resources. → [Roles](https://docs.nebius.com/iam/authorization/roles)

**Q3.** A CI/CD pipeline needs to create VMs programmatically. What account type should it use?
> A service account. Service accounts are designed for machine/automated access via CLI and API. → [Service accounts](https://docs.nebius.com/iam/service-accounts/manage)

**Q4.** Which key type does a service account use with Object Storage (S3-compatible)?
> Static keys (also called access keys for AWS-compatible APIs). → [Static keys](https://docs.nebius.com/iam/authorization/static-keys)

**Q5.** What algorithm does Nebius AI Cloud use for encryption at rest?
> AES-256. → [Encryption](https://docs.nebius.com/security/encryption)

**Q6.** You need to store an API key securely to use in multiple Slurm batch scripts. What Nebius service should you use?
> [MysteryBox](https://docs.nebius.com/mysterybox) — it stores secrets in encrypted form for reuse in pipelines and scripts.

**Q7.** Can you disable encryption on a shared filesystem?
> No. Shared filesystem encryption is always on and cannot be disabled. → [Storage types](https://docs.nebius.com/compute/storage/types#shared-filesystems)

**Q8.** You stop a GPU VM to save money but remain within quota limits. Will quotas be released?
> No. Quotas are only released when a VM is deleted. A stopped VM still occupies quotas. → [VM lifecycle](https://docs.nebius.com/compute/virtual-machines/lifecycle)

**Q9.** What is the default billing model for Nebius AI Cloud?
> [Pay-as-you-go (PAYG)](https://docs.nebius.com/signup-billing/billing-models/payg).

**Q10.** When does billing for a VM stop?
> When the deletion command is sent (not when deletion completes). → [VM lifecycle](https://docs.nebius.com/compute/virtual-machines/lifecycle)

---

## Domain 2 — Setting Up and Operating GPU Clusters

**Q11.** You want to create a GPU cluster with H200 GPUs in `eu-north1`. Which fabric should you select?
> `fabric-7` (H200 in eu-north1). → [InfiniBand fabrics](https://docs.nebius.com/compute/clusters/gpu)

**Q12.** What technology allows GPU-to-NIC data transfer without involving the CPU?
> GPUDirect RDMA. → [InfiniBand networking](https://docs.nebius.com/compute/clusters/gpu)

**Q13.** How does Nebius isolate InfiniBand traffic between different tenants using the same physical fabric?
> Using InfiniBand partition keys (P-Keys). Each GPU cluster gets a unique P-Key. → [InfiniBand security](https://docs.nebius.com/compute/clusters/gpu)

**Q14.** You try to add a VM to a GPU cluster but it fails. The VM is in a different project from the cluster. What is the problem?
> All VMs in a GPU cluster must be in the same project. → [GPU clusters](https://docs.nebius.com/compute/clusters/gpu)

**Q15.** What disk type should you use for a VM that needs high IOPS and reliability for a GlusterFS storage cluster?
> Network SSD IO M3 (`network_ssd_io_m3`). → [Storage types](https://docs.nebius.com/compute/storage/types)

**Q16.** What is the minimum disk size for Network SSD NRD?
> 93 GiB. The size must be a multiple of 93 GiB. → [Disk types](https://docs.nebius.com/compute/storage/types#disks)

**Q17.** A shared filesystem needs to be accessed by VMs in two different projects. Is this possible?
> No. A shared filesystem can only be attached to VMs in the same project. → [Shared filesystems](https://docs.nebius.com/compute/storage/types#shared-filesystems)

**Q18.** What CUDA driver preset gives you NVIDIA driver series 570.x?
> `cuda12.8`. → [GPU setup in Kubernetes](https://docs.nebius.com/kubernetes/gpu/set-up)

**Q19.** When updating the GPU driver preset on a Kubernetes node group, what happens to existing nodes?
> They are recreated according to the node group's deployment strategy (cordon → drain → delete → replace). → [Node groups](https://docs.nebius.com/kubernetes/node-groups/manage)

**Q20.** A VM enters `Error` status. What should you do?
> Delete the VM and create a new one. You cannot stop, start, or recover a VM in `Error` status. → [VM lifecycle](https://docs.nebius.com/compute/virtual-machines/lifecycle)

**Q21.** What is the total InfiniBand bandwidth for an 8-GPU H200 node?
> 3.2 Tbps (8 GPUs × 400 Gbps each). → [GPU types](https://docs.nebius.com/compute/virtual-machines/types)

**Q22.** Which Kubernetes operator is required when using NVIDIA B200 GPUs even without InfiniBand?
> NVIDIA Network Operator (required for B200 GPUs regardless of InfiniBand). → [GPU setup](https://docs.nebius.com/kubernetes/gpu/set-up)

---

## Domain 3 — Running Training and Inference Workloads

**Q23.** Where do Slurm nodes live in a Soperator cluster?
> As Kubernetes Pods. Soperator deploys Slurm onto a Kubernetes cluster. → [Soperator architecture](https://docs.nebius.com/slurm-soperator/overview/architecture)

**Q24.** A user connects to a Soperator login node and installs Python dependencies. Will those dependencies be available on worker nodes?
> Yes. All nodes share a root filesystem; changes on any node are immediately visible on all others. → [Soperator architecture](https://docs.nebius.com/slurm-soperator/overview/architecture)

**Q25.** What Slurm command submits a batch job?
> `sbatch my_job.sh` → [Running Slurm batch jobs](https://docs.nebius.com/slurm-soperator/jobs)

**Q26.** What does `--exclusive` do in an `sbatch` directive?
> It allocates all CPUs on the assigned worker nodes exclusively to this job, preventing other jobs from sharing those nodes. → [Slurm jobs](https://docs.nebius.com/slurm-soperator/jobs)

**Q27.** You want a training job to automatically restart if a worker node fails. Which `sbatch` directive do you add?
> `#SBATCH --requeue` → [Slurm jobs](https://docs.nebius.com/slurm-soperator/jobs)

**Q28.** A job script has `#SBATCH --time=1-00`. How long is the time limit?
> 1 day (24 hours). Format is `D-HH:MM:SS`; `1-00` means 1 day, 0 hours. → [Slurm jobs](https://docs.nebius.com/slurm-soperator/jobs)

**Q29.** What `sbatch` parameter sets job dependencies — e.g., job starts only after jobs 20 and 21 finish?
> `--dependency=afterany:20:21` → [Slurm jobs](https://docs.nebius.com/slurm-soperator/jobs)

**Q30.** What environment variable inside a running Slurm job contains the list of allocated worker nodes?
> `SLURM_JOB_NODELIST` → [Slurm jobs](https://docs.nebius.com/slurm-soperator/jobs)

**Q31.** Which orchestration model is better suited for inference serving with autoscaling?
> [Managed Kubernetes](https://docs.nebius.com/kubernetes) (Soperator/Slurm is better suited for batch distributed training).

**Q32.** How do you run a containerized workload in Soperator?
> Use `srun --container-image="<image>"` in the batch script. → [Jobs in containers](https://docs.nebius.com/slurm-soperator/jobs/containers/index)

---

## Domain 4 — Platform Automation and Maintenance

**Q33.** What Terraform command validates configuration syntax before applying?
> `terraform validate` → [Terraform quickstart](https://docs.nebius.com/terraform-provider/quickstart)

**Q34.** What is the source URL of the Nebius Terraform provider?
> `terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius` → [Terraform provider](https://docs.nebius.com/terraform-provider)

**Q35.** What Terraform resource creates a Kubernetes node group?
> `nebius_mk8s_v1_node_group` → [Terraform resource reference](https://docs.nebius.com/terraform-provider/reference/resources/mk8s_v1_node_group)

**Q36.** What field is required on every Nebius Terraform resource to specify which project it belongs to?
> `parent_id` → [Terraform quickstart](https://docs.nebius.com/terraform-provider/quickstart)

**Q37.** What happens when you run `terraform apply` and Terraform detects a node group needs a driver update?
> Nodes are recreated (existing nodes are cordoned, drained, deleted, then replacement nodes are created). → [Node groups](https://docs.nebius.com/kubernetes/node-groups/manage)

**Q38.** How do you store Terraform state remotely in Nebius?
> Store it in a Nebius Object Storage bucket. → [Storing Terraform state](https://docs.nebius.com/terraform-provider/store-terraform-state)

**Q39.** A GPU Kubernetes cluster was created before Dec 1, 2025. Are health checks enabled by default?
> No. Health checks are enabled by default only for clusters created on December 1, 2025 or later. → [Health checks](https://docs.nebius.com/kubernetes/maintenance/enable-disable-health-checks)

**Q40.** What observability tool would you use to set up alerts on GPU utilization thresholds?
> [Nebius AI Cloud Observability (alerts)](https://docs.nebius.com/observability/alerts), or integrate with [Grafana](https://docs.nebius.com/observability/metrics/grafana)/[Prometheus](https://docs.nebius.com/observability/metrics/prometheus).

**Q41.** What NCCL test is used to benchmark InfiniBand performance?
> The all-reduce NCCL test. → [NCCL tests in Kubernetes](https://docs.nebius.com/kubernetes/gpu/nccl-test) · [NCCL in Soperator](https://docs.nebius.com/slurm-soperator/jobs/examples/nccl-all-reduce)

**Q42.** What CLI command lists maintenance events?
> `nebius maintenance list` → [CLI reference](https://docs.nebius.com/cli/reference/maintenance)

**Q43.** What value of `on_preemption` is supported for preemptible VMs?
> `STOP` only. It stops the VM but does not delete or restart it. → [Preemptible VMs](https://docs.nebius.com/compute/virtual-machines/preemptible)

**Q44.** The Kubernetes Network Operator MOFED state shows `notReady`. How do you check the driver installation logs?
> `kubectl logs -n nvidia-network-operator $(kubectl get pods -n nvidia-network-operator | grep mofed | head -1 | awk '{print $1}')` → [GPU setup](https://docs.nebius.com/kubernetes/gpu/set-up)

---

# Quick Reference Cheat Sheet

```
═══════════════════════════════════════════════════════
IAM  docs.nebius.com/iam/overview
  Hierarchy:    Tenant → Projects → Resources
  Groups:       auditors < viewers < editors < admins
  SA keys:      authorized keys (CLI/API) | static keys (S3)
  Impersonate:  nebius compute ... -I <sa_id>

ENCRYPTION  docs.nebius.com/security/encryption
  Algorithm:    AES-256 via KMS (DEK + KEK)
  SSD:          Always encrypted (cannot disable)
  Filesystem:   Always encrypted (cannot disable)
  WEKA:         Optional, must set at creation

BILLING  docs.nebius.com/signup-billing/billing-models/overview
  Models:       PAYG (default) | Commitment discounts
  Quotas:       Only released on DELETE (not stop)
  Billing stop: When DELETE command is sent

═══════════════════════════════════════════════════════
GPU PLATFORMS  docs.nebius.com/compute/virtual-machines/types
  B300 SXM:   800 Gbps IB, 288 GB HBM3e  (uk-south1)
  B200 SXM:   400 Gbps IB, 180 GB HBM3e  (us-central1, me-west1)
  H200 SXM:   400 Gbps IB, 141 GB HBM3e  (eu-north1, eu-west1, us-central1)
  H100 SXM:   400 Gbps IB, 80 GB HBM3    (eu-north1)
  L40S PCIe:  No InfiniBand, 48 GB GDDR6  (eu-north1)

GPU CLUSTER  docs.nebius.com/compute/clusters/gpu
  Create:     nebius compute gpu-cluster create --infiniband-fabric <f>
  Isolation:  P-Keys per cluster (even on shared fabric)
  All VMs:    Must be in same project
  Test IB:    NCCL all-reduce test

VM LIFECYCLE  docs.nebius.com/compute/virtual-machines/lifecycle
  States:     Stopped → Starting → Running → Stopping → Stopped
  Error:      Delete and recreate
  Quotas:     Released only on DELETE

DISK TYPES  docs.nebius.com/compute/storage/types
  SSD:        network_ssd         | reliable | boot disks
  NRD:        network_ssd_nr      | fast     | K8s nodes, temp
  IO M3:      network_ssd_io_m3   | fast+reliable | GlusterFS, DB
  NRD/IO M3 size must be multiples of 93 GiB

STORAGE USE CASES
  Boot disks:          SSD
  K8s node disks:      SSD NRD
  DB/GlusterFS:        SSD IO M3
  Datasets:            Object Storage or Shared Filesystem
  Training checkpoints: Shared FS → Object Storage
  Inference weights:   Shared Filesystem

═══════════════════════════════════════════════════════
SLURM / SOPERATOR  docs.nebius.com/slurm-soperator
  Node types: login (sshd) | worker (slurmd) | controller (slurmctld)
  Shared FS:  All nodes share root / — no manual sync needed
  Submit:     sbatch --nodes=4 job.sh
  Multi-node: srun python train.py  (inside batch script)
  Status:     squeue | scontrol show job <id>
  Key flags:
    --nodes/-N            worker nodes
    --gpus-per-node       GPUs per node
    --exclusive           all CPUs, no sharing
    --requeue             auto-restart on failure
    --time/-t             time limit (01:00:00 or 1-00)
    --dependency/-d       job dependencies

KUBERNETES GPU SETUP  docs.nebius.com/kubernetes/gpu/set-up
  With built-in image:  --template-gpu-settings-drivers-preset cuda12.8
  Without image:        install GPU Operator (+ Network Operator if IB)
  Driver presets:       cuda12.8 (570.x) | cuda13.0 (580.x)
  Compatibility:        nebius mk8s node-group get-compatibility-matrix

═══════════════════════════════════════════════════════
TERRAFORM  docs.nebius.com/terraform-provider
  Source:     terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius
  Auth:       service_account block with env vars for key paths
  Workflow:   init → validate → plan → apply → destroy
  Parent ID:  required on every resource (= project ID)
  State:      docs.nebius.com/terraform-provider/store-terraform-state

OBSERVABILITY  docs.nebius.com/observability
  Entry point:  Observability Overview page in console
  Components:   Firing alerts | Metrics | Logs | Tracing
  Integrations: Grafana (docs.nebius.com/observability/metrics/grafana)
                Prometheus (docs.nebius.com/observability/metrics/prometheus)
  GPU health:   Enabled by default in K8s clusters (from Dec 1, 2025)
  NCCL test:    docs.nebius.com/kubernetes/gpu/nccl-test
  IB check:     kubectl get nicclusterpolicy... | jq '.status'
```

---

## Sources

- [IAM Overview](https://docs.nebius.com/iam/overview)
- [Managing Projects](https://docs.nebius.com/iam/manage-projects)
- [Service Accounts](https://docs.nebius.com/iam/service-accounts/manage)
- [Authorized Keys](https://docs.nebius.com/iam/service-accounts/authorized-keys)
- [Static Keys](https://docs.nebius.com/iam/authorization/static-keys)
- [Access Tokens](https://docs.nebius.com/iam/authorization/access-tokens)
- [IAM Roles and Groups](https://docs.nebius.com/iam/authorization/roles)
- [SSO with Microsoft Entra ID](https://docs.nebius.com/iam/federations/saml-sso)
- [Keycloak SSO](https://docs.nebius.com/iam/federations/configure-sso-keycloak)
- [JumpCloud SSO](https://docs.nebius.com/iam/federations/configure-sso-jumpcloud)
- [Encryption](https://docs.nebius.com/security/encryption)
- [Key Management Service](https://docs.nebius.com/kms)
- [Billing Models](https://docs.nebius.com/signup-billing/billing-models/overview)
- [PAYG](https://docs.nebius.com/signup-billing/billing-models/payg)
- [Commitment Discounts](https://docs.nebius.com/signup-billing/billing-models/committed-usage)
- [Billing Threshold](https://docs.nebius.com/signup-billing/payments/threshold)
- [VM Types and GPUs](https://docs.nebius.com/compute/virtual-machines/types)
- [VM Lifecycle](https://docs.nebius.com/compute/virtual-machines/lifecycle)
- [VM Maintenance](https://docs.nebius.com/compute/virtual-machines/maintenance)
- [Maintenance Reason Codes](https://docs.nebius.com/compute/virtual-machines/maintenance-reasons)
- [Preemptible VMs](https://docs.nebius.com/compute/virtual-machines/preemptible)
- [InfiniBand / GPU Clusters](https://docs.nebius.com/compute/clusters/gpu)
- [InfiniBand Topology](https://docs.nebius.com/compute/clusters/gpu/topology)
- [Storage Volume Types](https://docs.nebius.com/compute/storage/types)
- [Boot Disk Images](https://docs.nebius.com/compute/storage/boot-disk-images)
- [Automatic Security Updates](https://docs.nebius.com/compute/storage/automatic-updates)
- [Object Storage](https://docs.nebius.com/object-storage)
- [Kubernetes Overview](https://docs.nebius.com/kubernetes)
- [Kubernetes Quickstart](https://docs.nebius.com/kubernetes/quickstart)
- [Kubernetes Node Groups](https://docs.nebius.com/kubernetes/node-groups/manage)
- [Kubernetes Autoscaling](https://docs.nebius.com/kubernetes/node-groups/autoscaling)
- [Kubernetes GPU Setup](https://docs.nebius.com/kubernetes/gpu/set-up)
- [Kubernetes GPU Clusters (InfiniBand)](https://docs.nebius.com/kubernetes/gpu/clusters)
- [Topology-Aware Scheduling](https://docs.nebius.com/kubernetes/gpu/topology-aware-scheduling)
- [NCCL Tests in Kubernetes](https://docs.nebius.com/kubernetes/gpu/nccl-test)
- [Kubernetes Health Checks](https://docs.nebius.com/kubernetes/maintenance/enable-disable-health-checks)
- [Kubernetes Logs](https://docs.nebius.com/kubernetes/logs)
- [Soperator Overview](https://docs.nebius.com/slurm-soperator)
- [Soperator Architecture](https://docs.nebius.com/slurm-soperator/overview/architecture)
- [Why Slurm and Soperator](https://docs.nebius.com/slurm-soperator/overview/why-slurm-soperator)
- [Deployment Methods](https://docs.nebius.com/slurm-soperator/deploy/overview)
- [Managed Service for Soperator](https://docs.nebius.com/slurm-soperator/managed-soperator/manage)
- [Running Slurm Jobs](https://docs.nebius.com/slurm-soperator/jobs)
- [Jobs in Containers](https://docs.nebius.com/slurm-soperator/jobs/containers/index)
- [NCCL in Soperator](https://docs.nebius.com/slurm-soperator/jobs/examples/nccl-all-reduce)
- [Downloading Data in Soperator](https://docs.nebius.com/slurm-soperator/storage/download-data)
- [Serverless AI](https://docs.nebius.com/serverless/overview)
- [Serverless Endpoints](https://docs.nebius.com/serverless/endpoints/manage)
- [Terraform Provider](https://docs.nebius.com/terraform-provider)
- [Terraform Quickstart](https://docs.nebius.com/terraform-provider/quickstart)
- [Terraform Authentication](https://docs.nebius.com/terraform-provider/authentication)
- [Storing Terraform State](https://docs.nebius.com/terraform-provider/store-terraform-state)
- [Pulumi with Terraform Provider](https://docs.nebius.com/terraform-provider/pulumi)
- [Sensitive Values in Terraform](https://docs.nebius.com/terraform-provider/sensitive-values)
- [Observability Overview](https://docs.nebius.com/observability)
- [Alerts](https://docs.nebius.com/observability/alerts)
- [Metrics](https://docs.nebius.com/observability/monitoring)
- [Grafana Integration](https://docs.nebius.com/observability/metrics/grafana)
- [Prometheus Integration](https://docs.nebius.com/observability/metrics/prometheus)
- [Logs](https://docs.nebius.com/observability/logging)
- [Log Query Language](https://docs.nebius.com/observability/logs/query-language)
- [Tracing](https://docs.nebius.com/observability/traces)
- [Monitoring Agent](https://docs.nebius.com/observability/agents/monitoring-agent)
- [Nebius O11y Agent for Kubernetes](https://docs.nebius.com/observability/metrics/ingest/nebius-o11y-agent)
- [CLI Reference](https://docs.nebius.com/cli/reference)
- [CLI Quickstart](https://docs.nebius.com/cli/quickstart)
- [CLI Configure](https://docs.nebius.com/cli/configure)