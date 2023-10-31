# UGache

UGache is a fast mulit-GPU embedding cache that fully leverages fast interconnects (i.e. NVLink) between GPUs.
UGache uses a factored embedding extraction mechanism to improve the utilization of interconnets' bandwidth.
Furthermore, UGache proposes a MILP-based method to build a cache policy to balance the miss rate and global miss rate.

## Paper

[UGACHE: A Unified GPU Cache for Embedding-based Deep Learning](https://dl.acm.org/doi/10.1145/3600006.3613169) \
*Xiaoniu Song,Yiwen Zhang,Rong Chen,Haibo Chen* \
Proceedings of the 29th Symposium on Operating Systems Principles (SOSP'23)

## Project Structure

```
> tree .
├── coll_cache_lib              # source code of UGache (coll_cache is the internal name for ugache)
├── datagen                     # scripts to prepare datasets for GNN and DLR
│   ├── dlr
│   └── gnn
├── docker                      # docker file and scripts to pull and build ugache, dependency and other baselines
│   ├── Dockerfile.dlr
│   ├── Dockerfile.gnn
│   ├── setup_docker.dlr.sh
│   ├── setup_docker.gnn.sh
│   └── setup_docker.sh
├── example                     # minimal example to run UGache quickly
│   ├── dlr
│   └── gnn
├── eval                        # scripts for artifact evaluation (per-figure)
│   ├── dlr
│   └── gnn
└── python                      # source code of UGache
```

## Hardware Requirements
UGache aims to accelerate embedding access in multi-GPU platforms with NVLink support.
In this artifact, UGache only natively supports, and is evaluated on these 3 platforms:
- Server A with hard-wired NVLink
  - CPU: 2 x Intel Xeon Gold 6138 CPUs (20 cores each)
  - GPU: 4 x NVIDIA V100 (16GB) GPUs with symmetric fully-connected NVLink topology
- Server B with hard-wired NVLink
  - CPU: 2 x Intel Xeon Platinum 8163 CPUs (24 cores each)
  - GPU: 8 x NVIDIA V100 (32GB) GPUs with asymmetric NVLink topology, identical to DGX-1
- Server C with NVSwitch
  - CPU: 2 x Intel Xeon Gold 6348 CPU (28 cores each)
  - GPU: 8 x NVIDIA A100 (80GB) GPUs, connected via NVSwitch

These platforms are currently hard-coded in UGache.
For other undocumented multi-GPU platforms with NVLink interconnects, we provides preliminary support in this [document](platform-support.md).

## Setting up the Software Environment

We use NVIDIA's Merlin container as the base environment for UGache.
Most software dependencies have been prepared inside the image(e.g. PyTorch, TensorFlow, CUDA, cuDNN).
Please first ensure that your docker service supports CUDA.
Here is a [reference](https://stackoverflow.com/questions/59691207) to solve Docker building images with CUDA support.
Due to conflicts in dependencies, GNN and DLR evaluations are conducted in different containers(merlin-pytorch and merlin-tensorflow).


The docker images can be built using the following command:
```bash
cd <ugache-dir>/docker
docker build --pull -t ugache-gnn -f Dockerfile.gnn --build-arg RELEASE=false .
docker build --pull -t ugache-dlr -f Dockerfile.dlr --build-arg RELEASE=false .
```

Then launch containers using the following command:
```bash
docker run  --shm-size=200g --ulimit memlock=-1 --ulimit core=0 --runtime=nvidia --privileged=true --cap-add=SYS_ADMIN --cap-add=SYS_NICE --ipc=host --name ugache-ae-gnn -it ugache-gnn bash
docker run  --shm-size=200g --ulimit memlock=-1 --ulimit core=0 --runtime=nvidia --privileged=true --cap-add=SYS_ADMIN --cap-add=SYS_NICE --ipc=host --name ugache-ae-dlr -it ugache-dlr bash
```

Since datasets require a large disk volume, please also bind external storage to the container if your server stores large datasets on a separate device:
```bash
docker run  --shm-size=200g --ulimit memlock=-1 --ulimit core=0 --runtime=nvidia --privileged=true --cap-add=SYS_ADMIN --cap-add=SYS_NICE --ipc=host -v <extern_host_storage>:/dataset_gnn --name ugache-ae-gnn -it ugache-gnn bash
docker run  --shm-size=200g --ulimit memlock=-1 --ulimit core=0 --runtime=nvidia --privileged=true --cap-add=SYS_ADMIN --cap-add=SYS_NICE --ipc=host -v <extern_host_storage>:/dataset_dlr --name ugache-ae-dlr -it ugache-dlr bash
```

Then run these scripts to pull and build UGache, its dependencies and baselines:
```bash
## for GNN:
docker exec -it ugache-ae-gnn bash
bash /tmp/setup_docker.sh
bash /tmp/setup_docker.gnn.sh
## for DLR:
docker exec -it ugache-ae-dlr bash
bash /tmp/setup_docker.sh
bash /tmp/setup_docker.dlr.sh
```

Note that this will pull a new copy of UGache inside the container.
The UGache repository on host is no longer used.

### Prepare Gurobi license

UGache depends on Gurobi to solve MILP problems. Please refer to this [link](https://portal.gurobi.com/iam/licenses/request) to request a trial license.
We recommand `WLS Academic` license for academic users, ae it can be deployed in multiple containers.
Place the license file to `/opt/gurobi/gurobi.lic` in container, and verify that the license is properly installed using the following command:
```bash
/opt/gurobi-install/linux64/bin/gurobi_cl --license
```

## Quick Start

Please refer to [readme](example/README.md)

## Reproduce Evaluation Results

Please refer to [readme](eval/README.md)

## Contact
Contact xiaoniu.sxn [at] sjtu.edu.cn for assistance.