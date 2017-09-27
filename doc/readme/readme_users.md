# Installation

## Requirements

| Architecture | Operating system |
|:------------:|:------------------:|
| Intel x86_64 | CentOS 6.7 & 6.8 |
| Intel x86_64 | Ubuntu 16.04 |
| Intel x86_64 | Windows 7 |

## Prerequisites

| Operating system | CPU                          | GPU                |
|:-----------------|:----------------------------:|:------------------:|
| Linux            | Intel SDK for OpenCL v1.2    | AMD APP SDK v3.0   |
| Windows          | Intel SDK for OpenCL 2016 R3 | AMD APP SDK v3.0   |

Download links:
* [OpenCL Drivers and Runtimes for Intel Architecture](https://software.intel.com/en-us/articles/opencl-drivers)
* [APP SDK – A Complete Development Platform](http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/)

Other environments/configurations likely work as well, but are untested.

# Usage

## Basic
```zsh
./bin/ocladock_<type>_<N>wi -ffile <protein>.maps.fld -lfile <ligand>.pdbqt -nrun <nruns>
```
| Mandatory options | Description   | Value                     |
|:-----------------:|:-------------:|:-------------------------:|
| -ffile            |Protein file   |&lt;protein&gt;.maps.fld   |
| -lfile            |Ligand file    |&lt;ligand&gt;.pdbqt       |

## Example
```zsh
./bin/ocladock_amdgpu_64wi -ffile ./input/1stp/derived/1stp_protein.maps.fld -lfile ./input/1stp/derived/1stp_ligand.pdbqt -nrun 10

Kernel source file:                      ./device/calcenergy.cl                  
Kernel compilation flags:                 -I ./device -I ./common -DN64WI        

Executing docking runs:
        20%        40%       60%       80%       100%
---------+---------+---------+---------+---------+
**************************************************

Program run time 26.931 sec 
```

By default the output log file is written in the current working folder. Examples of output logs can be found under [examples/output](examples/output/).

## Supported arguments

| Argument | Description                  | Default value |
|:---------|:-----------------------------|--------------:|
| -nrun    | # Docking runs               | 1             |
| -nev     | # Energy evaluations         | 2500000       |
| -ngen    | # Generations                | 27000         |
| -lsit    | # Local-search iterations (max.) | 300       |
| -psize   | Population size              | 150           |
| -mrat    | Mutation rate                | 2 (%)         |
| -crat    | Crossover rate               | 80 (%)        |
| -lsrat   | Local-search rate            | 6 (%)         |
| -trat    | Tournament rate              | 60 (%)        |
| -resnam  | Name for docking output log  | "docking"     |

For a complete list of available arguments and their default values, check: [getparameters.cpp](host/src/getparameters.cpp).

## Images
Prebuilt images are provided for [Linux](prebuilt/linux) and [Windows](prebuilt/windows).

# Further reading
Go to [Index](doc/readme/home.md).
