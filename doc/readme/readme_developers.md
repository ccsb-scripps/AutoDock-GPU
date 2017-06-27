# File structure

## Main structure
```
.
├── bin
├── common
│   ├── calcenergy_basic.h
│   └── defines.h
├── device
│   ├── auxiliary_genetic.cl
│   ├── calcenergy.cl
│   ├── kernel1.cl
│   ├── kernel2.cl
│   ├── kernel3.cl
│   └── kernel4.cl
├── doc
│   ├── presentation
│   │   └── IWOCL2017_MolecularDocking_online_version.pdf
│   └── readme
│       ├── home.md
│       ├── readme_developers.md
│       └── readme_users.md
├── docking.dlg
├── docking.xml
├── examples
│   └── output
│       ├── ocladock_cpu_1stp_nrun100.dlg
│       ├── ocladock_cpu_3ce3_nrun100.dlg
│       ├── ocladock_gpu_1stp_nrun100.dlg
│       └── ocladock_gpu_3ce3_nrun100.dlg
├── host
│   ├── inc
│   │   ├── calcenergy.h
│   │   ├── getparameters.h
│   │   ├── miscellaneous.h
│   │   ├── performdocking.h
│   │   ├── processgrid.h
│   │   ├── processligand.h
│   │   └── processresult.h
│   └── src
│       ├── calcenergy.cpp
│       ├── getparameters.cpp
│       ├── main.cpp
│       ├── miscellaneous.cpp
│       ├── performdocking.cpp
│       ├── processgrid.cpp
│       ├── processligand.cpp
│       └── processresult.cpp
├── input
│   ├── 1stp
│   │   ├── 1STP.pdb
│   │   ├── BTN_600.gif.png
│   │   ├── derived
│   │   │   ├── 1stp.dlg
│   │   │   ├── 1stp.dpf
│   │   │   ├── 1stp.glg
│   │   │   ├── 1stp.gpf
│   │   │   ├── 1stp_ligand.pdb
│   │   │   ├── 1stp_ligand.pdbqt
│   │   │   ├── 1stp_nrun100.dpf
│   │   │   ├── 1stp_nrun10.dpf
│   │   │   ├── 1STP.pdb
│   │   │   ├── 1stp_protein.A.map
│   │   │   ├── 1stp_protein.C.map
│   │   │   ├── 1stp_protein.d.map
│   │   │   ├── 1stp_protein.e.map
│   │   │   ├── 1stp_protein.HD.map
│   │   │   ├── 1stp_protein.maps.fld
│   │   │   ├── 1stp_protein.maps.xyz
│   │   │   ├── 1stp_protein.N.map
│   │   │   ├── 1stp_protein.OA.map
│   │   │   ├── 1stp_protein.pdb
│   │   │   ├── 1stp_protein.pdbqt
│   │   │   └── 1stp_protein.SA.map
│   │   └── Ligands_noHydrogens_withMissing_1_Instances.sdf
│   └── 3ce3
│       ├── 3CE3.pdb
│       └── derived
│           ├── 3ce3.dlg
│           ├── 3ce3.dpf
│           ├── 3ce3.glg
│           ├── 3ce3.gpf
│           ├── 3ce3_ligand.pdb
│           ├── 3ce3_ligand.pdbqt
│           ├── 3ce3_nrun100.dpf
│           ├── 3ce3_nrun10.dpf
│           ├── 3CE3.pdb
│           ├── 3ce3_protein.A.map
│           ├── 3ce3_protein.C.map
│           ├── 3ce3_protein.d.map
│           ├── 3ce3_protein.e.map
│           ├── 3ce3_protein.F.map
│           ├── 3ce3_protein.HD.map
│           ├── 3ce3_protein.maps.fld
│           ├── 3ce3_protein.maps.xyz
│           ├── 3ce3_protein.N.map
│           ├── 3ce3_protein.OA.map
│           ├── 3ce3_protein.pdb
│           └── 3ce3_protein.pdbqt
├── LICENSE
├── logo.png
├── Makefile
├── prebuilt
│   ├── linux
│   │   ├── ocladock_cpu_16wi
│   │   ├── ocladock_cpu_32wi
│   │   ├── ocladock_cpu_64wi
│   │   ├── ocladock_gpu_16wi
│   │   ├── ocladock_gpu_32wi
│   │   └── ocladock_gpu_64wi
│   └── windows
│       ├── ocladock-win-cpu-16wi.exe
│       └── ocladock-win-gpu-64wi.exe
├── README.md
├── win
└── wrapcl
    ├── inc
    │   ├── BufferObjects.h
    │   ├── CommandQueues.h
    │   ├── commonMacros.h
    │   ├── Contexts.h
    │   ├── Devices.h
    │   ├── ImportBinary.h
    │   ├── ImportSource.h
    │   ├── Kernels.h
    │   ├── listAttributes.h
    │   ├── Platforms.h
    │   └── Programs.h
    └── src
        ├── BufferObjects.cpp
        ├── CommandQueues.cpp
        ├── Contexts.cpp
        ├── Devices.cpp
        ├── ImportBinary.cpp
        ├── ImportSource.cpp
        ├── Kernels.cpp
        ├── listAttributes.cpp
        ├── Platforms.cpp
        └── Programs.cpp

```

**[bin](bin)**: Linux binary files are placed here once compiled.

| File                     | Description                                                                   |
|--------------------------|-------------------------------------------------------------------------------|
| `ocladock_<type>_<N>wi`  | Binary file for `<type>` (cpu, gpu) device with `<N>` (16, 32, 64) work items |


**[common](common)**: common header files for host and device.

| File                | Description                                                                       |
|---------------------|-----------------------------------------------------------------------------------|
| [calcenergy_basic.h](common/calcenergy_basic.h)  | Basic defines and macros for energy calculation      |
| [defines.h](common/defines.h)           | Basic defines for energy calculation and kernels optimization |


**[device](device)**: source files containing kernels.

| File                 | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
| [auxiliary_genetic.cl](device/auxiliary_genetic.cl) | Auxiliary functions for energy calculation     |
| [calcenergy.cl](device/calcenergy.cl)   | Main function for energy calculation                       |
| [kernel1.cl](device/kernel1.cl) | `gpu_calc_initpop`: calculates the energy of initial population    |
| [kernel2.cl](device/kernel2.cl) | `gpu_sum_evals`: sums the evaluation counter states                |
| [kernel3.cl](device/kernel3.cl) | `perform_LS`: performs the local search                            |
| [kernel4.cl](device/kernel4.cl) | `gpu_gen_and_eval_newpops`: performs the genetic generation        |

**[doc](doc)**: documentation files.

**[host](host)**: host source files.

| File                 | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
| calcenergy{[.h](host/inc/calcenergy.h)}{[.cpp](host/src/calcenergy.cpp)} | Auxiliary functions for parallel energy-calculation     |
| getparameters{[.h](host/inc/getparameters.h)}{[.cpp](host/src/getparameters.cpp)}| Functions for processing program input arguments   |
| miscellaneous{[.h](host/inc/miscellaneous.h)}{[.cpp](host/src/miscellaneous.cpp)} | General-purpose functions    |
| [main.cpp](host/src/main.cpp) | Main source file     |
| performdocking{[.h](host/inc/performdocking.h)}{[.cpp](host/src/performdocking.cpp)} | Entry point for OpenCL-platform setup and kernels execution    |
| processgrid{[.h](host/inc/processgrid.h)}{[.cpp](host/src/processgrid.cpp)} | Functions for processing and converting the energy grids    |
| processligand{[.h](host/inc/processligand.h)}{[.cpp](host/src/processligand.cpp)} | Functions for performing operations in the ligand     |
| processresult{[.h](host/inc/processresult.h)}{[.cpp](host/src/processresult.cpp)} | Functions for processing docking results  |

**[input](input)**: chemical compounds taken from [PDB](http://www.rcsb.org/pdb/home/home.do).

| PDB complex                 | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
| [1stp](http://www.rcsb.org/pdb/explore/explore.do?structureId=1stp) | Structual origins of high-affinity biotin binding to streptavidin     |
| [3ce3](http://www.rcsb.org/pdb/explore/explore.do?structureId=3ce3) | Crystal structure of the tyrosine kinase domain of the hepatocyte growth factor receptor C-MET in complex with a Pyrrolopyridinepyridone based inhibitor     |


For both complexes, the containing folder has a similar structure. 

Input files were preprocessed before docking following the standard protocol using AutoDockTools.

**[input/1stp/derived](input/1stp/derived)**: 1stp input files 

| File                   | Description                                |
|----------------------|-----------------------------------------------------------------------------------|
| [1stp_ligand.pdb](input/1stp/derived/1stp_ligand.pdb)       | Ligand file in .pdb format                 |
| [1stp_ligand.pdbqt](input/1stp/derived/1stp_ligand.pdbqt)   | Ligand file in .pdbqt format (check [usage](doc/readme/readme_users.md))|
| [1stp_protein.A.map](input/1stp/derived/1stp_protein.A.map) | Affinity map for aromatic carbon           |
| [1stp_protein.C.map](input/1stp/derived/1stp_protein.C.map) | Affinity map for aliphatic carbon          |
| [1stp_protein.d.map](input/1stp/derived/1stp_protein.d.map) | Affinity map for desolvation               |
| [1stp_protein.e.map](input/1stp/derived/1stp_protein.e.map) | Affinity map for electrostatics            |
| [1stp_protein.HD.map](input/1stp/derived/1stp_protein.HD.map) | Affinity map for (donor) 1 H-bond hydrogen                            |
| [1stp_protein.maps.fld](input/1stp/derived/1stp_protein.maps.fld) | Grid map field file (check [usage](doc/readme/readme_users.md))   |
| [1stp_protein.maps.xyz](input/1stp/derived/1stp_protein.maps.xyz) |  Contains the minimum and maximum extends of the grid box in each dimension x, y, and z |
| [1stp_protein.N.map](input/1stp/derived/1stp_protein.N.map) | Affinity map for H-bonding nitrogen             |
| [1stp_protein.OA.map](input/1stp/derived/1stp_protein.OA.map) | Affinity map for (acceptor) 2 H-bonds oxygen  |
| [1stp_protein.pdb](input/1stp/derived/1stp_protein.pdb) | Protein filein .pdb format                          |
| [1stp_protein.pdbqt](input/1stp/derived/1stp_protein.pdbqt) | Protein filein .pdbqt format                    |
| [1stp_protein.SA.map](input/1stp/derived/1stp_protein.SA.map) | Affinity map for (acceptor) 2 H-bonds sulphur |
| [1stp.dlg](input/1stp/derived/1stp.dlg) | Docking log file when using original AutoDock program               |
| [1stp.dpf](input/1stp/derived/1stp.dpf) | AutoDock docking parameter file                                     |
| [1stp.glg](input/1stp/derived/1stp.glg) | Grid log file when using original AutoGrid program                  |
| [1stp.gpf](input/1stp/derived/1stp.gpf) | AutoDock grid parameter file                                        |


**[wrapcl](wrapcl)**: custom wrapper functions for OpenCL API calls (complaint to OpenCL 1.2).

| File                 | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
| BufferObjects{[.h](host/inc/BufferObjects.h)}{[.cpp](host/src/BufferObjects.cpp)} | Functions for allocating, mapping, copying, and querying info of buffers |
| CommandQueues{[.h](host/inc/CommandQueues.h)}{[.cpp](host/src/CommandQueues.cpp)} | Functions for creating and querying info of command queues     |
| [commonMacros.h](host/inc/commonMacros.h) | Header with defines to enable features and display of info of OpenCL elements (platform, devices, etc)     |
| Contexts{[.h](host/inc/Contexts.h)}{[.cpp](host/src/Contexts.cpp)} | Functions for creating and querying info of contexts    |
| Devices{[.h](host/inc/Devices.h)}{[.cpp](host/src/Devices.cpp)} | Functions for detecting available devices and querying their attributes     |
| ImportBinary{[.h](host/inc/ImportBinary.h)}{[.cpp](host/src/ImportBinary.cpp)} | Functions for loading kernel code and transforming (offline) it into device programs     |
| ImportSource{[.h](host/inc/ImportSource.h)}{[.cpp](host/src/ImportSource.cpp)} | Functions for loading kernel code and transforming (online) it into device programs    |
| Kernels{[.h](host/inc/Kernels.h)}{[.cpp](host/src/Kernels.cpp)} | Functions for setting kernel arguments, dispatching kernels, and querying kernel attributes     |
| listAttributes{[.h](host/inc/listAttributes.h)}{[.cpp](host/src/listAttributes.cpp)} | Definitions of OpenCL attributes    |
| Platforms{[.h](host/inc/Platforms.h)}{[.cpp](host/src/Platforms.cpp)} | Functions for detecting installed platforms and querying their attributes      |
| Programs{[.h](host/inc/Programs.h)}{[.cpp](host/src/Programs.cpp)} | Functions for querying program's info     |


# Compilation

## Basic
```zsh
make DEVICE=<TYPE> NUMWI=<NWI>
```

`<TYPE>` : CPU, GPU.

`<NUMWI>` : 16, 32, 64

After successful compilation, the host binary `ocladock_<type>_<N>wi` is placed under [bin](./bin).

`type` denotes the accelerator chosen: `cpu` or `gpu`.

`N` denotes the OpenCL work-group size: `16`, `32`, or `64`.

## All available options
```zsh
make DEVICE=<TYPE> NUMWI=<NWI> CONFIG=<CFG> DOCK_DEBUG=<Y/N> REPRO=<Y/N>
```
| Argument    | Description                           | Possible values          |
|-------------|---------------------------------------|--------------------------|
| DEVICE      | OpenCL device type                    | `<TYPE>`: CPU, GPU       |
| NUMWI       | Number of work items per work group   | `<NWI>` : 16, 32, 64     |
| CONFIG      | Host configuration                    | `<CFG>` : DEBUG, RELEASE |
| DOCK_DEBUG  | Enable debug info for host & device  | `<Y/N>` : YES, NO        |
| REPRO       | Reproduce results (remove randomness) | `<Y/N>` : YES, NO        |

## Configuration file
Check the configurations in the project's [Makefile](../Makefile).

# Installation

## Requirements
OCLADock is known to work in the following environments:

* Architecture: Intel x86_64
* Operating System: CentOS 6.7 & 6.8 / Ubuntu 16.04 / Windows 7

Other environments likely work as well, but are untested.
 
## Prerequisites

| Operating system | CPU                          | GPU                |
|:-----------------|:----------------------------:|:------------------:|
| Linux            | Intel SDK for OpenCL v1.2    | AMD APP SDK v3.0   |
| Windows          | Intel SDK for OpenCL 2016 R3 | AMD APP SDK v3.0   |


Other configurations likely work as well, but are untested.

## To keep in mind before compiling

**The corresponding environmental variables must be defined**
* CPU accelerator : `$(INTELOCLSDKROOT)`
* GPU accelerator : `$(AMDAPPSDKROOT)` 

**The corresponding paths for CPU/GPU drivers must be also defined**
* This is usually resolved automatically during SDK/driver installation
* In case it is not set, resolve it manually
    * E.g.: GPU accelerator on Linux: `/etc/ld.so.conf.d/amdgpu-pro-x86_64.conf` must contain the path of the driver

**Check vendor specific guidelines to setup both OpenCL platform correctly!**

# Further reading
Go to [Index](doc/readme/home.md).