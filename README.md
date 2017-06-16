OpenCL Accelerated Molecular Docking (OCLADock)
===============================================

[logo]

# Features

* OpenCL-accelerated version of AutoDock 4.2 (Lamarckian Genetic Algorithm).
* It targets platforms based on multi-core CPU and GPU accelerators.

# Setup
## Requirements
OCLADock is known to work in the following environment:

* Architecture: Intel x86_64
* Operating System: CentOS 6.7 & 6.8 / Ubuntu 16.04

## Prerequisites
* CPU:
	* Intel SDK for OpenCL v1.2
	* Intel OpenCL Runtime v16.1

* GPU
	* AMD APP SDK v3.0
	* AMDGPU-PRO v16.50

Other environments/configurations likely work as well, but are untested.

# Compilation

## Compilation on Linux
```zsh
make DEVICE=<TYPE> NUMWI=<NWI>
```
`<TYPE>` : CPU, GPU.

`<NUMWI>` : 16, 32, 64

After successful compilation, the host binary `ocladock_<type>_<N>wi` is placed under [bin](./bin).

`type` denotes the accelerator chosen: `cpu` or `gpu`.

`N` denotes the OpenCL work-group size: `16`, `32`, or `64`.

This can be configured in the [Makefile](Makefile).

## Compilation on Windows

Currently only binaries are distributed.

# Usage

## Basic command
```zsh
./bin/ocladock_<type>_<N>wi -ffile <protein>.maps.fld -lfile <ligand>.pdbqt -nrun <nruns>
```
Mandatory arguments:
* Protein file: `<protein>.maps.fld`
* Ligand file:  `<ligand>.pdbqt`

## Example
```zsh
./bin/ocladock_gpu_64wi -ffile ./input_data/1stp/derived/1stp_protein.maps.fld -lfile ./input_data/1stp/derived/1stp_ligand.pdbqt -nrun 10
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

# Documentation

For a complete documentation, check the [Documentation](doc/readme/home.md).

# Credits

Leonardo Solis-Vasquez and Andreas Koch. 2017. A Performance and Energy Evaluation of OpenCL-accelerated Molecular Docking. In Proceedings of the 5th International Workshop on OpenCL (IWOCL 2017). ACM, New York, NY, USA, Article 3, 11 pages. DOI: https://doi.org/10.1145/3078155.3078167

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
