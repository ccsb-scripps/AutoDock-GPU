This is the documentation for developers: source code structure, build instructions, tools required, etc.

# Source-code structure

## Main structure
```
.
+-- bin
+-- common
+-- device
+-- doc
+-- host
+-- input
+-- wrapcl


```

**bin**: binary files.

| File                     | Description                                                                   |
|--------------------------|-------------------------------------------------------------------------------|
| `ocladock_<type>_<N>wi`  | Binary file for `<type>` (cpu, gpu) device with `<N>` (16, 32, 64) work items |


**common**: common header files for host and device.

| File                | Description                                                                       |
|---------------------|-----------------------------------------------------------------------------------|
| [calcenergy_basic.h](common/calcenergy_basic.h)  | Basic defines and macros for energy calculation      |
| [defines.h](common/defines.h)           | Basic defines for energy calculation and kernels optimization |


**device**: source files containing kernels.

| File                 | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
| [auxiliary_genetic.cl](device/auxiliary_genetic.cl) | Auxiliary functions for energy calculation     |
| [calcenergy.cl](device/calcenergy.cl)   | Main function for energy calculation                       |
| [kernel1.cl](device/kernel1.cl) | `gpu_calc_initpop`: calculates the energy of initial population    |
| [kernel2.cl](device/kernel2.cl) | `gpu_sum_evals`: sums the evaluation counter states                |
| [kernel3.cl](device/kernel3.cl) | `perform_LS`: performs the local search                            |
| [kernel4.cl](device/kernel4.cl) | `gpu_gen_and_eval_newpops`: performs the genetic generation        |

**doc**: documentation files.

**host**: host source files.

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

**input**: chemical compounds taken from [PDB](http://www.rcsb.org/pdb/home/home.do).

| PDB complex                 | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
| [1stp](http://www.rcsb.org/pdb/explore/explore.do?structureId=1stp) | STRUCTURAL ORIGINS OF HIGH-AFFINITY BIOTIN BINDING TO STREPTAVIDIN     |
| [3ce3](http://www.rcsb.org/pdb/explore/explore.do?structureId=3ce3) | Crystal structure of the tyrosine kinase domain of the hepatocyte growth factor receptor C-MET in complex with a Pyrrolopyridinepyridone based inhibitor     |


For both complexes, the containing folder has a similar structure:


| File                 | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
|  |     |


**wrapcl**: custom wrapper functions for OpenCL API calls (complaint to OpenCL 1.2).

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
| DOCK_DEBUG  | Enable debug info from host & device  | `<Y/N>` : YES, NO        |
| REPRO       | Reproduce results (remove randomness) | `<Y/N>` : YES, NO        |

## Configuration file
Check the configurations in the project's [Makefile](../Makefile).

# Requirements
