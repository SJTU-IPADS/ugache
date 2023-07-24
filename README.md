# Artifact Evaluation for UGache [SOSP'23]

This repository contains scripts and instructions for reproducing the experiments in our SOSP'23 paper "UGACHE: A Unified GPU Cache for Embedding-based Deep Learning Systems".

> **NOTE:** For artifact evaluation committee, please directly jump to [Reproducing the Results](#reproducing-the-results) section and reproduce the results on the provided server. For other readers, please follow the instructions below to setup the environment and reproduce the results on your own server.

## Table of Contents

[TOC]

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
│   ├── setup_docker.sh
├── eval                        # evaluation scripts (per-figure)
│   ├── dlr
│   └── gnn
└── python                      # source code of UGache
```

## Hardware Requirements

UGache natively supports, and is evaluated on these 3 platforms:
- Server A with hard-wired NVLink
  - CPU: 2 x Intel Xeon Gold 6138 CPUs (20 cores each)
  - GPU: 4 x NVIDIA V100 (16GB) GPUs with symmetric fully-connected NVLink topology
- Server B with hard-wired NVLink
  - CPU: 2 x Intel Xeon Platinum 8163 CPUs (24 cores each)
  - GPU: 8 x NVIDIA V100 (32GB) GPUs with asymmetric NVLink topology, identical to DGX-1
- Server C with NVSwitch
  - CPU: 2 x Intel Xeon Gold 6348 CPU (28 cores each)
  - GPU: 8 x NVIDIA A100 (80GB) GPUs, connected via NVSwitch

- Software Requirements
  - xxx

## Setting up the Environment

We use NVIDIA's Merlin container as base environment for UGache.
Due to conflicts in dependencies, GNN and DLR evaluations are conducted in different containers(merlin-pytorch and merlin-tensorflow).

<!-- ## Installation on Bare Metal Server
We provide an one-click script to setup the environment on bare metal server. The script requires sudo permission and will install the required packages and Brainstorm itself. -->

The docker images can be built by the following command:
```bash
cd <ugache-dir>/docker
docker build --pull -t ugache-gnn -f Dockerfile.gnn --build-arg RELEASE=false .
docker build --pull -t ugache-dlr -f Dockerfile.dlr --build-arg RELEASE=false .
```

Then launch containers by the following command:
```bash
# fixme: run in background rather than interactive
docker run  --shm-size=200g --ulimit memlock=-1 --ulimit core=0 --runtime=nvidia --privileged=true --cap-add=SYS_ADMIN --cap-add=SYS_NICE --ipc=host --name ugache-ae-gnn -it ugache-gnn bash
docker run  --shm-size=200g --ulimit memlock=-1 --ulimit core=0 --runtime=nvidia --privileged=true --cap-add=SYS_ADMIN --cap-add=SYS_NICE --ipc=host --name ugache-ae-dlr -it ugache-dlr bash
```

Please also bind external storage into the container if your server stores large datasets on separate device:
```bash
docker run  --shm-size=200g --ulimit memlock=-1 --ulimit core=0 --runtime=nvidia --privileged=true --cap-add=SYS_ADMIN --cap-add=SYS_NICE --ipc=host -v <extern_host_storage>:/dataset_gnn --name ugache-ae-gnn -it ugache-gnn bash
docker run  --shm-size=200g --ulimit memlock=-1 --ulimit core=0 --runtime=nvidia --privileged=true --cap-add=SYS_ADMIN --cap-add=SYS_NICE --ipc=host -v <extern_host_storage>:/dataset_dlr --name ugache-ae-dlr -it ugache-dlr bash
```

We also provide an online image on github registry. The image can be run by the following command:

```bash
xxx
```

### Prepare Gurobi license

UGache depends on gurobi to solve MILP problem. Please refer to link xxx to request a trial license.
Verify license is properly installed by the following command:
```bash
gurobi_cl --license
```

# Preparing the Dataset:
This script will download the dataset. Later, we will provide a detailed description for preparing the dataset.

```bash
xxx
```

# Reproducing the Results

Please enter the directory `xxx` and run the following commands to reproduce the results. Each scripts will run the corresponding experiment, visualize the results.

```bash
xxx
```

## Reproducing the Results on xx Server (xx time cost)

We provide a one-click script to reproduce the results on xx server. This script includes the reproduction of  Figure-xx to xx.
```bash
xx
```

User can also reproduce the results of each figure separately by running the corresponding script. The following table shows the corresponding script for each figure.

|  Figure   |       Script       | Description                                                                                   |
| :-------: | :----------------: | --------------------------------------------------------------------------------------------- |
| Figure-xx | `xx`               | xxx                                                                                           |

