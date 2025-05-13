# Google Cloud NVIDIA B200测试手册

以下指令以两个节点共16个GPU为例，对于128个GPU做相应修改即可。

```shell
cd ~
git clone https://github.com/totorochina/gpu-poc.git
```

## 一、创建资源预留
创建方式参考，\
https://cloud.google.com/compute/docs/instances/create-single-project-future-reservations#specify-vm-properties

```shell
# to be updated
export FUTURE_RESERVATION_NAME=gpu-poc-b200-2nodes
export MACHINE_TYPE=a4-highgpu-8g
export START_TIME="2025-05-19T09:00:00+08:00"
export END_TIME="2025-05-20T00:00:00+08:00"
export TOTAL_COUNT=2
export REGION=us-central1
export ZONE=us-central1-b

gcloud beta compute future-reservations create FUTURE_RESERVATION_NAME \
    --auto-delete-auto-created-reservations \
    --machine-type=MACHINE_TYPE \
    --start-time=START_TIME \
    --end-time=END_TIME \
    --total-count=TOTAL_COUNT \
    --zone=ZONE
```

预留创建后，我们会找TAM帮忙审批，并在测试日之前通过。


## 二、创建GKE集群
创建方式参考，\
https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute

使用Cluster Toolkit创建GKE集群，Cluster Toolkit使用terraform并以官方最佳实践的方式一键部署和销毁集群。\
它会自动创建的内容有。\
- 1、VPC中需要设置的RDMA CX-7网络
- 2、经过参数优化的共享存储gcsfuse，并设置为pv/pvc供容器直接使用
- 3、fio I/O测试yaml，以及jobset模版yaml等帮助你快速上手的脚手架和说明

```shell
# to be updated
export BUCKET_NAME=gcs-gpu-poc-b200

cd ~
git clone https://github.com/GoogleCloudPlatform/cluster-toolkit.git

cd cluster-toolkit && git checkout main && make

# create bucket for terraform
gcloud storage buckets create gs://BUCKET_NAME \
    --default-storage-class=STANDARD \
    --location=REGION \
    --uniform-bucket-level-access
gcloud storage buckets update gs://BUCKET_NAME --versioning
```

更新 [examples/gke-a4/gke-a4-deployment.yaml](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/gke-a4/gke-a4-deployment.yaml)中需要填写的字段。
一个参考个例子如[gke-a4-deployment.yaml](examples/gke-a4-deployment.yaml)所示。

### 创建集群
```shell
./gcluster deploy -d \
examples/gke-a4/gke-a4-deployment.yaml \
examples/gke-a4/gke-a4.yaml --auto-approve
```

集群创建完毕后，会输出相关使用说明和提示，内容类似于，

```shell
instructions_a4-cluster = <<EOT
The following networks have been authorized to access this cluster:
  kubectl-access-network: 0.0.0.0/0"

To add authorized networks you can allowlist your IP with this command:
  gcloud container clusters update gke-a4-hzchen-poc \
    --region us-central1 \
    --project gpu-launchpad-playground \
    --enable-master-authorized-networks \
    --master-authorized-networks <IP Address>/32

Use the following command to fetch credentials for the created cluster:
  gcloud container clusters get-credentials gke-a4-hzchen-poc \
    --region us-central1 \
    --project gpu-launchpad-playground

Use the following Kubernetes Service Account in the default namespace to run your workloads:
  workload-identity-k8s-sa
The GCP Service Account mapped to this Kubernetes Service Account is:
  gke-a4-hzchen-poc-gke-wl-sa@gpu-launchpad-playground.iam.gserviceaccount.com
EOT
instructions_fio-bench-job-template = <<EOT
A GKE job file has been created locally at:
  /Users/hzchen/scripts/cluster-toolkit/gke-a4-hzchen-poc/primary/my-job-540f.yaml

Use the following commands to:
Submit your job:
  kubectl create -f /Users/hzchen/scripts/cluster-toolkit/gke-a4-hzchen-poc/primary/my-job-540f.yaml

EOT
instructions_job-template = <<EOT
A GKE job file has been created locally at:
  /Users/hzchen/scripts/cluster-toolkit/gke-a4-hzchen-poc/primary/run-nvidia-smi-ce3d.yaml

Use the following commands to:
Submit your job:
  kubectl create -f /Users/hzchen/scripts/cluster-toolkit/gke-a4-hzchen-poc/primary/run-nvidia-smi-ce3d.yaml

EOT
Collecting terraform outputs from gke-a4-hzchen-poc/primary
Writing outputs artifact from deployment group primary to file gke-a4-hzchen-poc/.ghpc/artifacts/primary_outputs.tfvars

###############################
Find instructions for cleanly destroying infrastructure and advanced manual
deployment instructions at:

gke-a4-hzchen-poc/instructions.txt
```

后续使用只需完成身份认证
```shell
gcloud container clusters get-credentials gke-a4-poc \
    --region us-central1 \
    --project tx-poc-250507

gcloud config set container/cluster gke-a4-poc

```
即可使用kubectl进行后续操作
```
kubectl get all
kubectl get nodes
```

## 三、NCCL测试
测试方式参考，https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom#deploy-run-nccl-test

官方有提供一个yaml，包含所需的pod/service定义以及安装配置了所有所需驱动/变量的镜像以及nccl测试脚本。
```shell
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/refs/heads/master/gpudirect-rdma/nccl-test-a4.yaml
```
部署完毕后，即可通过以下方式登陆到该pod，像vm一样进行后续测试。
```shell
kubectl exec nccl-test-host-1 -it -- /scripts/container_entry.sh shell
```

对于NCCL的测试，我们有准备相应的脚本，方便在多个节点之间进行测试。

```shell
# global allreduce
kubectl exec nccl-test-host-1 -it -- /usr/local/gib/scripts/run_nccl_tests.sh -t all_reduce -b 1K -e 8G nccl-host-1 nccl-host-2

# multi allreduce
kubectl exec nccl-test-host-1 -it -- /usr/local/gib/scripts/run_nccl_tests.sh -t all_reduce -b 1K -e 8G -m 0x7 nccl-host-1 nccl-host-2
kubectl exec nccl-test-host-1 -it -- /usr/local/gib/scripts/run_nccl_tests.sh -t all_reduce -b 1K -e 8G -m 0x1 nccl-host-1 nccl-host-2

# all2all
kubectl exec nccl-test-host-1 -it -- /usr/local/gib/scripts/run_nccl_tests.sh -t alltoall -b 1K -e 8G nccl-host-1 nccl-host-2
```

对于更多的节点，可以考虑用jobset的方式执行。

```shell
# update num nodes before use.
kubectl apply -f examples/nccl-jobset-test.yaml
```

## 四、Nemo测试

[gpu-recipe](https://github.com/AI-Hypercomputer/gpu-recipes/tree/main)是GCP官方用于复现公开benchmark数据的repo。
这里将复用该repo跑Nemo pretraining。

```shell
# check your queue name from `kubectl get queue`
export KUEUE_NAME=a4

git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
```

通过helm提交llama 3.1 70B的pretraining的任务。以下为2节点16 GPUs版本。\
这里指定了之前用于存放terraform states的存储桶，此时兼用于存放DLLogger日志。

```shell
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file nemo_config=~/gpu-poc/examples/llama3-1-70b-16gpus-a4-bf16.yaml \
    --set workload.image=us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo25.02-gib1.0.5-A4 \
    --set network.hostNetwork="false" \
    --set "volumes.gcsMounts[0].bucketName=${BUCKET_NAME}" \
    --set queue=${KUEUE_NAME} \
    --set workload.arguments="{trainer.max_steps=50}" \
    --set workload.gpus=16 \
    $USER-llama-3-1-70b-nemo-bf16 \
    $REPO_ROOT/src/helm-charts/a4/nemo-training
```

对于16节点128 GPUs，--set-file nemo_config的参数可以改为这个yaml，
```shell
$REPO_ROOT/src/frameworks/a4/nemo-configs/llama3-1-70b-256gpus-a4-bf16.yaml
```

对于mistral 8x7B，可以使用这个yaml
```shell
$REPO_ROOT/src/frameworks/a4/nemo-configs/mixtral-8x7b-16-32-gpus-a4-bf16.yaml
$REPO_ROOT/src/frameworks/a4/nemo-configs/mixtral-8x7b-256gpus-a4-bf16.yaml
```

查看训练滚动日志
```shell
kubectl get all
kubectl logs -f job.batch/hzchen-llama-3-1-70b-nemo-bf16
```

Nemo框架可集成DLLogger写训练日志，查看DLLogger日志并使用脚本计算相关指标。
```shell
gcloud storage cp -r gs://hzchen-poc-gpu/nemo-experiments/megatron_gpt/hzchen-llama-3-1-70b-nemo-bf16-1746783738-zv6u/ .
python -u examples/stat_dllogger.py --file ./hzchen-llama-3-1-70b-nemo-bf16-1746783738-zv6u/dllogger/rank-0/dllogger.json --warm_up 10 --num_gpus 16 --accelerator b200 --precision bf16 --model llama3-70b -v
```

完成后可删除helm chart
```shell
helm list
helm delete hzchen-llama-3-1-70b-nemo-bf16
```

## 五、销毁资源

```shell
cd ~/cluster-toolkit
./gcluster destroy ./gke-a4-poc --auto-approve
```

如果第二天还要继续测试，无需销毁集群，仅讲GPU节点组缩为0即可。
```shell
gcloud container clusters resize gke-a4-poc --node-pool=a4-highgpu-8g-a4-pool --num-nodes=0  --location=us-central1 --quiet
```

**备注：\
步骤一、二可以交由合作伙伴代办，用户直接从集群认证开始。**