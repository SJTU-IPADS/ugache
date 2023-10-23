# Quick Start

We provide examples of GNN training and DLR inference.
Note that the phase of loading correct model parameter values from disk is skipped for fast reproduction.

## GNN Training Example

Here we provide examples training GraphSAGE on the Products dataset using UGache + DGL + PyTorch.

### Prepare Dataset

In UGache's GNN container, run the following commands to download and process a small GNN datasets, OGBN-Products:
```bash
cd /ugache/datagen/gnn
python products.py
```

By default, the datasets will be placed in `/datasets_gnn`:

```bash
tree /datasets_gnn -L 2
/datasets_gnn
├── data-raw
│   ├── products
│   └── products.zip
├── gnnlab
│   └── products-undir
└── wholegraph
    └── ogbn_products
```

### Start Training

We provide two examples to perform supervised training of GraphSAGE on the Products dataset.
These two scripts both use UGache for embedding extraction and DGL+PyTorch for training.
The difference between the two scripts is that they respectively use DGL and WholeGraph to conduct graph sampling.

```bash
cd /ugache/example/gnn
# use DGL for graph sampling
python dgl_sample.py
# use WholeGraph for graph sampling
python wg_sample.py
```

## DLR Inference Example

Here we provide an example to serve DLRM model inference with a synthetic dataset generated using zipfian distribution.
The scripts uses UGache for embedding extraction and TensorFlow as backend for inference.
Note that in DLR inference with tensorflow, UGache adopts similar API like HugeCTR Hierarchical Parameter Server.
A json config [file](dlr/config.json) is required for configuration.

```bash
cd /ugache/example/dlr
python main.py
```
