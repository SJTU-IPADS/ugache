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

## Setting up the Environment

We use NVIDIA's Merlin container as base environment for UGache.
Please first make sure that your docker service supports CUDA.
Here is a [reference](https://stackoverflow.com/questions/59691207) to solve Docker building images with CUDA support.
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

Since datasets requires large disk volume, please also bind external storage into the container if your server stores large datasets on separate device:
```bash
docker run  --shm-size=200g --ulimit memlock=-1 --ulimit core=0 --runtime=nvidia --privileged=true --cap-add=SYS_ADMIN --cap-add=SYS_NICE --ipc=host -v <extern_host_storage>:/dataset_gnn --name ugache-ae-gnn -it ugache-gnn bash
docker run  --shm-size=200g --ulimit memlock=-1 --ulimit core=0 --runtime=nvidia --privileged=true --cap-add=SYS_ADMIN --cap-add=SYS_NICE --ipc=host -v <extern_host_storage>:/dataset_dlr --name ugache-ae-dlr -it ugache-dlr bash
```

Then run these scripts to pull and build UGache, dependencies and baselines:
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


### Prepare Gurobi license

UGache depends on gurobi to solve MILP problem. Please refer to [link](https://portal.gurobi.com/iam/licenses/request) to request a trial license.
We recommand `WLS Academic` license for academic users, since it can be deployed in multiple container.
Place the license file to `/opt/gurobi/gurobi.lic` in container, and verify license is properly installed by the following command:
```bash
/opt/gurobi-install/linux64/bin/gurobi_cl --license
```

## Preparing the Dataset:
We provide scripts to prepare DLR and GNN datasets in `datagen`.

### GNN Datasets
By default, GNN datasets will be placed in `/datasets_gnn`:
```bash
tree /datasets_gnn -L 2
/datasets_gnn
├── data-raw                     # original downloaded dataset
│   ├── com-friendster
│   ├── com-friendster.tar.zst
│   ├── mag240m_kddcup2021
│   ├── mag240m_kddcup2021.zip
│   ├── papers100M-bin
│   └── papers100M-bin.zip
├── gnnlab                       # converted dataset for GNNLab
│   ├── com-friendster
│   ├── mag240m-homo
│   └── papers100M-undir
└── wholegraph                   # converted dataset for UGache and WholeGraph
    ├── com_friendster
    ├── mag240m_homo
    └── ogbn_papers100M
```

Run the following commands to download and process GNN datasets:
```bash
# run following commands in GNN container
cd /ugache/datagen/gnn
python friendster.py
python mag240M.py
python papers100M.py
```
### DLR Datasets
By default, DLR datasets will be placed in `/datasets_dlr`:
```bash
tree /datasets_dlr -L 2
/datasets_dlr
├── data-raw                     # original downloaded dataset
│   ├── crite_tb
└── processed
    ├── criteo_tb
    └── syn_a12_s100_c800m
```

Since there's no permanent url to download criteo TB dataset, please download it manually from [ailab.criteo.com](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset) or [aliyun](https://tianchi.aliyun.com/dataset/144736), and place `day_0.gz` ~ `day_23.gz` under `/datasets_dlr/data-raw/criteo_tb/.
Then, Run the following commands to process DLR datasets:
```bash
# run following commands in GNN container
cd /ugache/datagen/dlr
python syn.py
cd criteo
bash criteo.sh
```

## Reproducing the Results
Our experiments have been automated by scripts. Each figure in our paper is treated as one experiment and is associated with a subdirectory in `ugache/eval`. The script will automatically run the experiment, save the logs into files, parse the output data from the files, and plot corresponding figure.

```bash
tree /ugache/eval -L 2
/ugache/eval
├── dlr
│   ├── common
│   └── figurexx
└── gnn
    ├── common
    ├── common_gnnlab
    ├── figure11-4v100
    ├── figure11-8a100
    ├── figure11-8a100-fix-cache-rate
    ├── figure11-8v100
    ├── figure12-4v100
    ├── figure12-8a100
    ├── figure12-8a100-fix-cache-rate
    ├── figure12-8v100
    ├── figure13
    ├── figure14
    └── figure15
```

In each `figurexx` folder, execute following commands. Take figure13 for exmaple:
```bash
# evals in gnn folder should be run in gnn container
cd /ugache/eval/gnn/figure13
make run
make plot
```

The `run` command runs all tests, and logs will be saved to `run-logs` folder.
The `plot` command will parse logs in `run-logs` folder, and plot corresponding figure to `data.eps`.

> We recommand the `eps-preview` extension in vscode to quickly preview eps figures.

We also provide original log files used in our paper submission in `run-logs-paper` folder.
You may `make plot-paper` to directly plot figures using these log files to quickly reproduce figures in paper before actually run all tests.
