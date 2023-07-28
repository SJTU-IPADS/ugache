# Artifact Evaluation for UGache [SOSP'23]

This repository contains scripts and instructions for reproducing the experiments in our SOSP'23 paper "UGACHE: A Unified GPU Cache for Embedding-based Deep Learning Systems".

> **NOTE:** For artifact evaluation committee, please directly jump to [Reproducing the Results](#reproducing-the-results) section and reproduce the results on the provided server. For other readers, please follow the instructions below to setup the environment and reproduce the results on your own server.

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
UGache aims to speed up embedding access in multi-GPU platform with NVLink support.
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

Later we will provide a detailed description of how to support other multi-GPU platforms with NVLink.

## Setting up the Software Environment

We use NVIDIA's Merlin container as base environment for UGache.
Most software dependencies has been prepared inside the image(e.g. PyTorch, TensorFlow, CUDA, cuDNN).
Please first make sure that your docker service supports CUDA.
Here is a [reference](https://stackoverflow.com/questions/59691207) to solve Docker building images with CUDA support.
Due to conflicts in dependencies, GNN and DLR evaluations are conducted in different containers(merlin-pytorch and merlin-tensorflow).


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
Due to limited time and disk volume, the preprocessing of embeddings is not included.
In this artifact, the embeddings are initialized without loading correct embedding values from dataset.
This only affects the numerical correctness of training/inference, and does not affect the computation workflow.

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
Apart from downloading ~300GB raw data, the preprocess may take around 1 hour.
The final dataset in `gnnlab` and `wholegraph` occupies 130GB, while the `data-raw` directory occupies up to 600GB.

### DLR Datasets
By default, DLR datasets will be placed in `/datasets_dlr`:
```bash
tree /datasets_dlr -L 2
/datasets_dlr
├── data-raw                     # original downloaded dataset
│   ├── criteo_tb
└── processed                    # converted dataset
    ├── criteo_tb
    └── syn_a12_s100_c800m
```

Since there's no permanent url to download criteo TB dataset, please download it manually from [ailab.criteo.com](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset) or [aliyun](https://tianchi.aliyun.com/dataset/144736), and place `day_0.gz` ~ `day_23.gz` under `/datasets_dlr/data-raw/criteo_tb/.
Then, Run the following commands to process DLR datasets:
```bash
# run following commands in DLR container
cd /ugache/datagen/dlr
python syn.py
cd criteo
bash criteo.sh
```
Depending on your network, downloading and preprocessing full criteo TB dataset may take up to 24 hours and around 2TB disk volume. The final dataset in `processed` occupies 700GB.

## Reproducing the Results
Our experiments have been automated by scripts. Each figure in our paper is treated as one experiment and is associated with a subdirectory in `ugache/eval`. The script will automatically run the experiment, save the logs into files, parse the output data from the files, and plot corresponding figure.

```bash
tree /ugache/eval -L 2
/ugache/eval
├── dlr
│   ├── common
│   ├── figure11-4v100
│   ├── figure11-8a100
│   ├── figure11-8v100
│   ├── figure12-4v100
│   ├── figure12-8a100
│   ├── figure12-8v100
│   ├── figure16
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
The additional `gnn/figure11-8a100-fix-cache-rate` and `gnn/figure12-8a100-fix-cache-rate` is for the purpose of fixing mis-configured cache rate.
On server C, GNNLab, Rep_U and UGache should be able to cache all embeddings of Papers100M and Com-Friendster, since the GPU has 80GB memory.
We accidentally used a smaller cache rate during submission.

### Rreproducing all experiments
We provide a one-click script to reproduce the results on multi-gpu server.
These scripts simply chain commands in following "Reproducing single figure" section.

```bash
$ cd /ugache/eval/gnn        # evals in gnn folder should be run in gnn container
                             # for dlr, enter /ugache/eva/dlr in dlr container
$ bash run-all-4v100.sh      # run scripts that match the platform: run-all-(4v100,8v100,8a100).sh
```

### Reproducing single figure
In each `figure*` folder, execute following commands. Take `gnn/figure13`` for exmaple:
```bash
# evals in gnn folder should be run in gnn container
$ cd /ugache/eval/gnn/figure13
$ make run
$ make plot
$ ls data*
data.dat	data.eps
$ cat data.dat
short_app	policy_impl	dataset_short	epoch_e2e_time
sage_unsup	Cliq	PA	5.137098
sage_unsup	Coll	PA	4.433856
sage_unsup	Cliq	CF	7.197985
sage_unsup	Coll	CF	6.035764
sage_unsup	Cliq	MAG	22.249095
sage_unsup	Coll	MAG	22.575602
sage_sup	Cliq	PA	0.663522
sage_sup	Coll	PA	0.572377
sage_sup	Cliq	CF	0.922484
sage_sup	Coll	CF	0.804951
sage_sup	Cliq	MAG	1.585167
sage_sup	Coll	MAG	1.389422
sage_unsup	GNNLab	PA	8.6040
sage_unsup	GNNLab	CF	14.1925
sage_unsup	GNNLab	MAG	47.0257
sage_sup	GNNLab	PA	1.0986
sage_sup	GNNLab	CF	2.0982
sage_sup	GNNLab	MAG	3.0826
```

The `make run` command runs all tests, and logs will be saved to `run-logs` folder.
The `make plot` command will first parse logs in `run-logs` folder to produce a `data.dat` file, then plot corresponding figure to `data.eps`.

> We recommand the `eps-preview` extension in vscode to quickly preview eps figures.

We also provide original log files used in our paper submission in `run-logs-paper` folder.
You may `make plot-paper` to directly plot figures using these log files to quickly reproduce figures in paper before actually run all tests.

Each figure should be evaluated on designated platform. The following table shows the platform and estimated time for each figure:
|       Figure       | Platform | Estimated Time |
| :----------------- | :------: | -------------: |
| dlr/figure11-4v100 | Server A |     30 min     |
| dlr/figure11-8v100 | Server B |     30 min     |
| dlr/figure11-8a100 | Server C |     30 min     |
| dlr/figure12-4v100 | Server A |     30 min     |
| dlr/figure12-8v100 | Server B |     30 min     |
| dlr/figure12-8a100 | Server C |     20 min     |
| dlr/figure16       | Server C |     10 min     |
| gnn/figure11-4v100 | Server A |     60 min     |
| gnn/figure11-8v100 | Server B |     60 min     |
| gnn/figure11-8a100 | Server C |     30 min     |
| gnn/figure12-4v100 | Server A |     50 min     |
| gnn/figure12-8v100 | Server B |     50 min     |
| gnn/figure12-8a100 | Server C |     30 min     |
| gnn/figure13       | Server C |     70 min     |
| gnn/figure14       | Server C |     40 min     |
| gnn/figure15       | Server C |      0 min     |

> **Note**: Due to the inability to access server B and server C, we provide the screencasts for the results on these platforms.
> The table below shows our experimental results after screen recording.

|  Server  | APP |                                              Screencast                                             |                                           Log & Script Archive                                            |                md5                 |
| :------: | :-: | :-------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: | :--------------------------------: |
| Server B | GNN | [gnn-8v100.mp4](https://ipads.se.sjtu.edu.cn:1313/d/640a7ad8121648c1a90f/files/?p=%2Fgnn-8v100.mp4) | [gnn-8v100.tar.gz](https://ipads.se.sjtu.edu.cn:1313/d/640a7ad8121648c1a90f/files/?p=%2Fgnn-8v100.tar.gz) | `d7b73603be17d5168363cf9810322b7c` |
| Server B | DLR | [dlr-8v100.mp4](https://ipads.se.sjtu.edu.cn:1313/d/640a7ad8121648c1a90f/files/?p=%2Fdlr-8v100.mp4) | [dlr-8v100.tar.gz](https://ipads.se.sjtu.edu.cn:1313/d/640a7ad8121648c1a90f/files/?p=%2Fdlr-8v100.tar.gz) | `67e35f264a63be94ef23d2676375879c` |
| Server C | GNN | [gnn-8a100.mp4](https://ipads.se.sjtu.edu.cn:1313/d/640a7ad8121648c1a90f/files/?p=%2Fgnn-8a100.mp4) | [gnn-8a100.tar.gz](https://ipads.se.sjtu.edu.cn:1313/d/640a7ad8121648c1a90f/files/?p=%2Fgnn-8a100.tar.gz) | `6269cb2fbef963314d5a468e83798973` |
| Server C | DLR | [dlr-8a100.mp4](https://ipads.se.sjtu.edu.cn:1313/d/640a7ad8121648c1a90f/files/?p=%2Fdlr-8a100.mp4) | [dlr-8a100.tar.gz](https://ipads.se.sjtu.edu.cn:1313/d/640a7ad8121648c1a90f/files/?p=%2Fdlr-8a100.tar.gz) | `8fc5ebcc8a6a1a02d4a248c6b2326817` |

> In the screeencast, we will first display the branch information of the code repository, then start the experiment using a one-click script.
> The script will delete all previours `run-logs` first.
> After running all experiments, the entire directory is compressed, with its corresponding md5 value printed.
> Reviewers can use this value to verify consistency between the provided tar and the one in the screen recording, and run `make plot` in each evaluted figure to plot figures and examine results..