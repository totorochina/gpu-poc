# POC Runtime Helm Chart

This Helm chart deploys a configurable number of NCCL nodes for GPU testing and development.

## Introduction

This chart creates a set of pods and headless services with names following the pattern `{RELEASE-NAME}-{NUMBER}`. Each pod runs the specified container image and includes necessary configurations for GPU workloads.

## Installing the Chart



```bash
export CHECKPOINT_PVC=$(kubectl get pvc -o json | jq -r '.items[] | select(.metadata.name | test("^checkpoint-.*")) | .metadata.name')
export TRAINING_PVC=$(kubectl get pvc -o json | jq -r '.items[] | select(.metadata.name | test("^training-.*")) | .metadata.name')
export N_NODES=2

cat <<EOF >custom-values.yaml
# Number of NCCL nodes to create
replicaCount: ${N_NODES}

# Persistent Volume Claims
volumes:
  persistentVolumeClaims:
    checkpoint:
      claimName: ${CHECKPOINT_PVC}
    training:
      claimName: ${TRAINING_PVC}
EOF

helm install gpu ./examples/poc-runtime -f custom-values.yaml
```

## Examples

### Example 1: Overwriting specific values

```bash
helm install gpu ./examples/poc-runtime -f custom-values.yaml \
  --set replicaCount=1
```

### Example 2: Scaling node counts
```bash
helm delete gpu
helm install gpu ./examples/poc-runtime -f custom-values.yaml \
  --set replicaCount=16
```
