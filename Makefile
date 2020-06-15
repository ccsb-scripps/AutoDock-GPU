# AutoDock-GPU Makefile

# ------------------------------------------------------
# Note that environment variables must be defined
# before compiling
# DEVICE?
# if DEVICE=CPU: CPU_INCLUDE_PATH?, CPU_LIBRARY_PATH?
# if DEVICE=GPU: GPU_INCLUDE_PATH?, GPU_LIBRARY_PATH?
#
# Cuda will be automatically detect and used as the
# when these conditions are true:
#      - DEVICE=CUDA *XOR*
#      - DEVICE=GPU *AND*
#      - nvcc exists *AND*
#      - GPU_INCLUDE_PATH =<path to Cuda includes> *AND*
#      - GPU_LIBRARRY_PATH=<path to Cuda libraries> *AND*
#      - a test code (including cuda_runtime_api.h,
#        and linking to -lcuda and -lcudart) is
#        succesfully compiled
# in any other case, OpenCL will be used
# OpenCL GPU path can be explicitly used with
# DEVICE=OCLGPU
# ------------------------------------------------------
# Choose OpenCL device
# Valid values: CPU, GPU, CUDA, OCLGPU

ifeq ($(DEVICE), $(filter $(DEVICE),GPU CUDA))
TEST_CUDA := $(shell ./test_cuda.sh nvcc "$(GPU_INCLUDE_PATH)" "$(GPU_LIBRARY_PATH)")
# if user specifies DEVICE=CUDA it will be used (wether the test succeeds or not)
# if user specifies DEVICE=GPU the test result determines wether CUDA will be used or not
ifeq ($(DEVICE)$(TEST_CUDA),GPUyes)
override DEVICE:=CUDA
endif
endif
ifeq ($(DEVICE),CUDA)
override DEVICE:=GPU
export
include Makefile.Cuda
else
ifeq ($(DEVICE),OCLGPU)
override DEVICE:=GPU
export
endif
include Makefile.OpenCL
endif
