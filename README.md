# UGache

This is the artifact for the paper "UGACHE: A Unified GPU Cache for Embedding-based Deep Learning Systems", We are going to reproduce the main results of the paper in this artifact.

> **NOTE:** For artifact evaluation committee, please directly jump to [Reproducing the Results](#reproducing-the-results) section and reproduce the results on the provided server. For other readers, please follow the instructions below to setup the environment and reproduce the results on your own server.

# Repository Structure

- `xxx`: xxx

# Hardware and Software Requirements

- Hardware Requirements
  - Server with hard-wired NVLink
    - CPU: xxx
    - GPU: NVIDIA V100 (16GB) GPU x 4

- Software Requirements
  - xxx

# Setting up the Environment

We provide two options to set up the environment: bare metal server and docker container.

## Installation on Bare Metal Server

We provide an one-click script to setup the environment on bare metal server. The script requires sudo permission and will install the required packages and Brainstorm itself.

```bash
xxx
```

## Installation with Docker Container

### Building the Docker Image

We also provide a docker image to setup the environment. The docker image can be built by the following command:

```bash
xxx
```

### Starting the Docker Container

The docker image can be run by the following command:

```bash
xxx
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

