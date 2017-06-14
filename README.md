OpenCL Accelerated Molecular Docking (OCLADock)
===============================================

[logo]

# Features

* OpenCL-accelerated version of the Lamarckian Genetic Algorithm of AutoDock 4.2.
* It targets platforms based on multi-core CPU and GPU accelerators.

# Setup
## Requirements
OCLADock is known to work in the following environment:

* Intel x86_64 architecture
* CentOS 6.7 & 6.8 / Ubuntu 16.04

## Prerequisites
* CPU Driver: Intel OpenCL Runtime v16.1
* GPU Driver: AMDGPU-PRO v16.50
* CPU SDK: Intel SDK for OpenCL v1.2
* GPU SDK: AMD APP SDK v3.0

Other environments/configurations likely work as well, but are untested.

# Compilation

## Compilation on Linux
```zsh
make DEVICE=<TYPE>
```
The valid values for <TYPE> are: CPU or GPU.

After successful compilation, the host binary `ocladock_<N>wi` is placed in the project root-folder.

`N` denotes the OpenCL work-group size. This can be configured in the [Makefile](Makefile).

## Compilation on Windows

# Usage

## Basic command
```zsh
./ocladock -ffile <protein>.maps.fld -lfile <ligand>.pdbqt -nrun <nruns>
```
Mandatory arguments:
* Protein file: `-ffile <protein>.maps.fld`
* Ligand file:  `-lfile <ligand>.pdbqt`

## Example
```zsh
./ocladock_64wi -ffile ./input_data/1stp/derived/1stp_protein.maps.fld -lfile ./input_data/1stp/derived/1stp_ligand.pdbqt -nrun 10
```
By default the output log file is written in the root folder: [docking.dlg](docking.dlg)

## Supported arguments
For a complete list of available arguments and their default values, check: [getparameters.cpp](host/src/getparameters.cpp)

| Argument | Description                  | Default value |
|----------|------------------------------|---------------|
| -nrun    | # Docking runs               | 1             |
| -nev     | # Energy evaluations         | 2500000       |
| -ngen    | # Generations                | 27000         |
| -lsit    | # Local-search iterations (max.) | 300       |
| -psize   | Population size              | 150           |
| -mrat    | Mutation rate                | 2 (%)         |
| -crat    | Crossover rate               | 80 (%)        |
| -lsrat   | Local-search rate            | 6 (%)         |
| -trat    | Tournament rate              | 60 (%)        |
| -resname | Name for docking output log  | "docking"     |

# License

# Credits

Leonardo Solis-Vasquez and Andreas Koch. 2017. A Performance and Energy Evaluation of OpenCL-accelerated Molecular Docking. In Proceedings of the 5th International Workshop on OpenCL (IWOCL 2017). ACM, New York, NY, USA, Article 3, 11 pages. DOI: https://doi.org/10.1145/3078155.3078167
