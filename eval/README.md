# Artifact Evaluation

## Preparing the Dataset:
We provide scripts to prepare DLR and GNN datasets in the `datagen` folder.
Due to limited time and disk space, the preprocessing of embeddings is not included.
In this artifact, the embeddings are initialized without loading the correct embedding values from the dataset.
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
The final datasets in `gnnlab` and `wholegraph` occupy 130GB, while the `data-raw` directory occupies up to 600GB.

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

Since there's no permanent url to download criteo TB dataset, please download it manually from [ailab.criteo.com](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset) or [aliyun](https://tianchi.aliyun.com/dataset/144736), and place `day_0.gz` ~ `day_23.gz` under `/datasets_dlr/data-raw/criteo_tb/`.
Then, run the following commands to process DLR datasets:
```bash
# run following commands in DLR container
cd /ugache/datagen/dlr
python syn.py
cd criteo
bash criteo.sh
```
Depending on your network, downloading and preprocessing full criteo TB dataset may take up to 24 hours and consume around 2TB disk volume. The final dataset in `processed` occupies 700GB.

## Reproducing the Results
Our experiments have been automated using scripts. Each figure in our paper is considered as one experiment and is associated with a subdirectory in `ugache/eval`. The script will automatically run the experiment, save the logs into files, parse the output data from the files, and plot corresponding figure.

```bash
tree /ugache/eval -L 2
/ugache/eval
├── dlr
│   ├── figure11-4v100
│   ├── figure11-8a100
│   ├── figure11-8v100
│   ├── figure12-4v100
│   ├── figure12-8a100
│   ├── figure12-8v100
│   ├── figure16
└── gnn
    ├── figure11-4v100
    ├── figure11-8a100
    ├── figure11-8v100
    ├── figure12-4v100
    ├── figure12-8a100
    ├── figure12-8v100
    ├── figure13
    ├── figure14
    └── figure15
```

### Rreproducing all experiments
We provide a one-click script to reproduce the results on multi-gpu server.
These scripts simply chain commands in the following "Reproducing single figure" section.

```bash
$ cd /ugache/eval/gnn        # GNN tests in gnn folder should be run in gnn container
                             # for DLR tests, enter /ugache/eva/dlr in dlr container
$ bash run-all-4v100.sh      # run scripts that match the platform: run-all-(4v100,8v100,8a100).sh
```

### Reproducing a single figure
In each `figure*` folder, execute the following commands. Take `dlr/figure11-4v100` for exmaple:

```bash
# tests in dlr folder should be run in dlr container
$ cd /ugache/eval/dlr/figure11-4v100
$ make run
$ make plot
$ ls data*
data.dat	data.eps
$ cat data.dat
short_app	policy_impl	dataset_short	step.train
dlrm	SOK	CR	0.005778
dlrm	HPS	CR	0.004299
dlrm	UGache	CR	0.002626
dcn	SOK	CR	0.007870
dcn	HPS	CR	0.006381
dcn	UGache	CR	0.004722
dlrm	SOK	SYN	0.014536
dlrm	HPS	SYN	0.018224
dlrm	UGache	SYN	0.008524
dcn	SOK	SYN	0.047721
dcn	HPS	SYN	0.046759
dcn	UGache	SYN	0.037482
```

The `make run` command runs all tests, and logs will be saved to the `run-logs` folder.
The `make plot` command will first parse logs in `run-logs` folder to produce a `data.dat` file, then plot corresponding figure to `data.eps`.

### Reproducing a single run

Each figure folder containers a `runner.py` file, and the `make run` is simply an alias of `python runner.py`.
The python script iterates all configurations, generates a command for each configuration and runs it via `os.system`.
You may execute `python runner.py -m` to see what command it generates and manually run one configuration.
We recommand the `eps-preview` extension in vscode to quickly preview eps figures.

### Plot using provided log file

We also provide original log files used in our paper submission in `run-logs-paper` folder.
You may run `make plot-paper` to directly plot figures using these log files to quickly reproduce the figures in paper without running all tests.

### Miscellaneous

The method UGache uses to dedicate different cores to run different kernel may not work on all driver version.
Note that manually run a configuration requires launching [NVIDIA-MPS service](https://docs.nvidia.com/deploy/mps/index.html#topic_5_1) separately.