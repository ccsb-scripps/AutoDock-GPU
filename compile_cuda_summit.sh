#!/bin/bash

# CUDAV=11.0.3
 CUDAV=10.1.243

 module load gcc/8.1.0 cuda/${CUDAV}
 
 export GPU_LIBRARY_PATH=/sw/summit/cuda/${CUDAV}/lib64
 export GPU_INCLUDE_PATH=/sw/summit/cuda/${CUDAV}/include

 make DEVICE=CUDA OVERLAP=ON 

