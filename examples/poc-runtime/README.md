# POC Runtime Helm Chart

This Helm chart deploys a configurable number of NCCL nodes for GPU testing and development.

## Introduction

This chart creates a set of pods and headless services with names following the pattern `nccl-host-${NUMBER}`. Each pod runs the specified container image and includes necessary configurations for GPU workloads.

## Prerequisites

- Kubernetes cluster with GPU support
- Helm 3.0+
- Persistent volume claims for checkpoint and training data
- Proper network configuration for RDMA interfaces

## Installing the Chart

To install the chart with the release name `my-release`:

```bash
helm install my-release ./examples/poc-runtime
```

## Configuration

The following table lists the configurable parameters of the chart and their default values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of NCCL nodes to create | `2` |
| `images.container.repository` | Container image repository | `nvcr.io/nvidia/pytorch` |
| `images.container.tag` | Container image tag | `25.01-py3` |
| `images.container.pullPolicy` | Container image pull policy | `IfNotPresent` |
| `images.initContainer.repository` | Init container image repository | `us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic` |
| `images.initContainer.tag` | Init container image tag | `v1.0.5` |
| `images.initContainer.pullPolicy` | Init container image pull policy | `Always` |
| `resources.requests.cpu` | CPU requests | `150m` |
| `resources.limits.nvidia.com/gpu` | GPU limits | `8` |
| `volumes.sharedMemory.sizeLimit` | Shared memory size limit | `250Gi` |
| `volumes.persistentVolumeClaims.checkpoint.claimName` | Checkpoint PVC name | `checkpoint-gke-a4-hzchen-poc-4b214df7-pvc` |
| `volumes.persistentVolumeClaims.training.claimName` | Training PVC name | `training-gke-a4-hzchen-poc-8d5e8c21-pvc` |

## Example: Installing with Custom Values

Create a custom values file (e.g., `custom-values.yaml`):

```yaml
replicaCount: 4
images:
  container:
    tag: 24.12-py3
resources:
  limits:
    nvidia.com/gpu: 4
```

Then install the chart with:

```bash
helm install my-release ./examples/poc-runtime -f custom-values.yaml
```

## Example: Updating Node Count

To change the number of nodes for an existing deployment:

```bash
helm upgrade my-release ./examples/poc-runtime --set replicaCount=8
```

## Example: Updating Container Images

To update the container image for an existing deployment:

```bash
helm upgrade my-release ./examples/poc-runtime --set images.container.tag=24.12-py3
```
