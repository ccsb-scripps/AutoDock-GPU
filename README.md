AutoDock-GPU: AutoDock for GPUs using OpenCL
============================================

<img src=".png" width="200">

# Features

* OpenCL-accelerated version of AutoDock4.2.6. It leverages its embarrasingly parallelizable LGA by processing ligand-receptor poses in parallel over multiple compute units.
* Besides the legacy Solis-Wets local search method, AutoDock-GPU adds newly implemented local-search methods based on gradients of the scoring function. One of these methods, ADADELTA, has proven to increase significantly the docking quality in terms of RMSDs and scores.
* It targets platforms based on GPU as well as multicore CPU accelerators.
* Observed speedups of up to 4x (quad-core CPU) and 56x (GPU) over the original serial AutoDock 4.2 (Solis-Wets) on CPU.

# Setup

| Operating system                         | CPU                          |GPU                                  |
|:----------------------------------------:|:----------------------------:|:-----------------------------------:|
|CentOS 6.7 & 6.8 / Ubuntu 14.04 & 16.04   | Intel SDK for OpenCL 2017    | AMD APP SDK v3.0 / CUDA v8.0 & v9.0 |
|macOS Catalina 10.15.1                    | Apple / Intel                | Apple / Intel Iris, Radeon Vega 64  |


Other environments or configurations likely work as well, but are untested.

# Compilation

```zsh
make DEVICE=<TYPE> NUMWI=<NWI>
```

| Parameters | Description            | Values                                             |
|:----------:|:----------------------:|:--------------------------------------------------:|
| `<TYPE>`   | Accelerator chosen     | `CPU`, `GPU`                                       |
| `<NWI>`    | OpenCL work-group size | `1`, `2`, `4`, `8`, `16`, `32`, `64`, `128`, `256` |

Hints: The best work-group size depends on the GPU and workload. Try `NUMWI=128` or `NUMWI=64` for modern cards with the example workloads. On macOS, use `NUMWI=1` for CPUs.

After successful compilation, the host binary **autodock_&lt;type&gt;_&lt;N&gt;wi** is placed under [bin](./bin).

| Binary-name portion | Description            | Values                                            |
|:-------------------:|:----------------------:|:-------------------------------------------------:|
| **&lt;type&gt;**    | Accelerator chosen     | `cpu`, `gpu`                                      |
| **&lt;N&gt;**       | OpenCL work-group size | `1`, `2`, `4`, `8`,`16`, `32`, `64`, `128`, `256` |


# Usage

## Basic command
```zsh
./bin/autodock_<type>_<N>wi \
-ffile <protein>.maps.fld \
-lfile <ligand>.pdbqt \
-nrun <nruns>
```

| Mandatory options | Description   | Value                     |
|:-----------------:|:-------------:|:-------------------------:|
| -ffile            |Protein file   |&lt;protein&gt;.maps.fld   |
| -lfile            |Ligand file    |&lt;ligand&gt;.pdbqt       |

## Example
```zsh
./bin/autodock_gpu_64wi \
-ffile ./input/1stp/derived/1stp_protein.maps.fld \
-lfile ./input/1stp/derived/1stp_ligand.pdbqt \
-nrun 10
```
By default the output log file is written in the current working folder. Examples of output logs can be found under [examples/output](examples/output/).

## Supported arguments

| Argument | Description                                           | Default value    |
|:---------|:------------------------------------------------------|-----------------:|
| -nrun    | # LGA runs                                            | 1                |
| -nev     | # Score evaluations (max.) per LGA run                | 2500000          |
| -ngen    | # Generations (max.) per LGA run                      | 27000            |
| -lsmet   | Local-search method                                   | sw (Solis-Wets)  |
| -lsit    | # Local-search iterations (max.)                      | 300              |
| -psize   | Population size                                       | 150              |
| -mrat    | Mutation rate                                         | 2 (%)            |
| -crat    | Crossover rate                                        | 80 (%)           |
| -lsrat   | Local-search rate                                     | 80 (%)           |
| -trat    | Tournament (selection) rate                           | 60 (%)           |
| -resnam  | Name for docking output log                           | _"docking"_      |
| -hsym    | Handle symmetry in RMSD calc.                         | 1                |
| -devnum  | OpenCL device number                                  | 1                |
| -cgmaps  | Use individual maps for CG-G0 instead of the same one | 0 (use same map) |

For a complete list of available arguments and their default values, check [getparameters.cpp](host/src/getparameters.cpp).

# Documentation

Visit the project [Wiki](https://github.com/ccsb-scripps/AutoDock-GPU/wiki).
