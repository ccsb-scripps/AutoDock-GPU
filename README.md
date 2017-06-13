OpenCL Accelerated Molecular Docking (OCLADock)
===============================================

[logo]

# Features

* OpenCL-accelerated version of the Lamarckian Genetic Algorithm of AutoDock 4.2.
* It targets platforms based on multi-core CPUs and GPUs accelerators.

# Setup
## Requirements
OCLADock is known to work in the following environment:

* PPPPP
* LLLLLL
* UUUUU

## Prerequisites
* OpenCL Intel SDK
* Intel driver
* AMD APP SDK
* AMD GPU driver

Other environments/configurations likely work as well, but are untested.

# Compilation

## Compilation on Linux
```zsh
make
```
After successful compilation, the host binary `ocladock_<N>wi` is placed in the project root-folder.

`N` denotes the OpenCL work-group size. This can be configured in the [Makefile][Makefile]

## Compilation on Windows

# Usage

## Basic command
```zsh
./ocladock -ffile <protein>.maps.fld -lfile <ligand>.pdbqt -nrun <nruns>
```

## Example
```zsh
./ocladock_64wi -ffile ./input_data/1stp/derived/1stp_protein.maps.fld -lfile ./input_data/1stp/derived/1stp_ligand.pdbqt -nrun 10
```

## Supported arguments
For a complete list of available arguments and their default values, check: [getparameters.cpp][host/src/getparameters.cpp].

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
