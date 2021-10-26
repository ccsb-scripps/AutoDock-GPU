#!/bin/bash

 module use /sw/summit/modulefiles/ums/stf010/Core
 module load llvm/14.0.0-latest cuda

 export GPU_PATH=${OLCF_CUDA_ROOT}

 make DEVICE=OMPGPU COMPILER=llvm NUMWI=64

