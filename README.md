OpenCL Accelerated Molecular Docking (OCLADock)
===============================================

<img src="logo.png" width="200">

# Features

* OpenCL-accelerated version of AutoDock 4.2 running a Lamarckian Genetic Algorithm (LGA)
* It leverages the LGA paralelism, as entities of multiple docking runs are computed simultaneously
* It targets platforms based on multi-core CPU and GPU accelerators
* Observed speedups of up to 4x (quad-core CPU) and 56x (GPU) over the original serial AutoDock 4.2 on CPU.

# Easy Download

If you are not familiar with Gitlab, the easiest way to download all of OCLADock (source code, prebuilt-binaries for Linux and Windows, sample input data) is to
use the Download icon (close to the top right of this webpage, just above the list of files) and use a familiar archive format (e.g., .zip) to fetch everything at once.

# Setup
## Requirements
OCLADock is known to work in the following environments:

| Architecture | Operating System |
|:------------:|:------------------:|
| Intel x86_64 | CentOS 6.7 & 6.8 |
| Intel x86_64 | Ubuntu 16.04 |
| Intel x86_64 | Windows 7 |

Other environments likely work as well, but are untested.

## Prerequisites

| Operating system | CPU                          | GPU                |
|:-----------------|:----------------------------:|:------------------:|
| Linux            | Intel SDK for OpenCL v1.2    | AMD APP SDK v3.0   |
| Windows          | Intel SDK for OpenCL 2016 R3 | AMD APP SDK v3.0   |

Other configurations likely work as well, but are untested.

# Ready-to-run Executables

We provide ready-to-run executables for [Linux](prebuilt/linux) and [Windows](prebuilt/windows). 
These executables have been compiled with a fixed number of work-items (**_wi_**, basically the degree of parallel processing done). If you are not sure, 
you should start with the versions using 16 work-items on a quad-core CPU and 64 work-items for a GPU. These values gave the 
best performance on our target platforms. The best values for your CPU or GPU might be different.

# Compilation

You only need to do this if you want to target our sources to a different system or modify the code. This can be configured in the [Makefile](Makefile).

## Compilation on Linux
```zsh
make DEVICE=<TYPE> NUMWI=<NWI>
```

| Parameters | Description            | Values       |
|:-----------|:----------------------:|:------------:|
| `<TYPE>`   | Accelerator chosen     | `CPU`, `GPU`     |
| `<NWI>`    | OpenCL work-group size | `16`, `32`, `64`   |


After successful compilation, the host binary `ocladock_<type>_<N>wi` is placed under [bin](./bin).

| Binary-name portion | Description   | Values            |
|:-----------|:----------------------:|:-----------------:|
| `<type>`   | Accelerator chosen     | `cpu`, `gpu`      |
| `<N>`      | OpenCL work-group size | `16`, `32`, `64`  |


## Compilation on Windows

A Microsoft Visual Studio 2013 solution for two configurations **_ocladock-cpu-deb_** and **_ocladock-gpu-deb_** can be found in the [win](win/) folder.

# Usage

## Basic command
```zsh
./bin/ocladock_<type>_<N>wi -ffile <protein>.maps.fld -lfile <ligand>.pdbqt -nrun <nruns>
```

| Mandatory options | Description   | Value               |
|:-----------------:|:-------------:|:-------------------:|
| -ffile            |Protein file   |<protein>.maps.fld   |
| -lfile            |Ligand file    |<ligand>.pdbqt       |

## Example
```zsh
./bin/ocladock_gpu_64wi -ffile ./input/1stp/derived/1stp_protein.maps.fld -lfile ./input/1stp/derived/1stp_ligand.pdbqt -nrun 10
```
By default the output log file is written in the current working folder. 

Examples of output logs can be found under [examples/output](examples/output/).

## Supported arguments

| Argument | Description                      | Default value |
|:---------|:---------------------------------|--------------:|
| -nrun    | # Docking runs                   | 1             |
| -nev     | # Energy evaluations             | 2500000       |
| -ngen    | # Generations                    | 27000         |
| -lsit    | # Local-search iterations (max.) | 300           |
| -psize   | Population size                  | 150           |
| -mrat    | Mutation rate                    | 2 (%)         |
| -crat    | Crossover rate                   | 80 (%)        |
| -lsrat   | Local-search rate                | 6 (%)         |
| -trat    | Tournament rate                  | 60 (%)        |
| -resnam  | Name for docking output log      | "docking"     |

For a complete list of available arguments and their default values, check: [getparameters.cpp](host/src/getparameters.cpp).

## Images
Prebuilt images are provided for [Linux](prebuilt/linux) and [Windows](prebuilt/windows).

# Documentation

For a complete documentation, check the [Documentation](doc/readme/home.md).

# Bibliographic information for citing OCLADock

Leonardo Solis-Vasquez and Andreas Koch. 2017. A Performance and Energy Evaluation of OpenCL-accelerated Molecular Docking. In Proceedings of the 5th International Workshop on OpenCL (IWOCL 2017). ACM, New York, NY, USA, Article 3, 11 pages. DOI: https://doi.org/10.1145/3078155.3078167

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
