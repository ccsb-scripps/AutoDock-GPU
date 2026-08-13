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
# Valid values: CPU, GPU, CUDA, OCLGPU, OPENCL

OVERLAP = ON

# HIP/ROCm backend: native AMD GPU path ported from the CUDA kernels. Selected
# explicitly with DEVICE=HIP; bypasses the nvcc CUDA probe entirely.
ifeq ($(DEVICE), HIP)
override DEVICE:=GPU
export
$(info Using HIP)
$(info )
include Makefile.Hip
else

ifeq ($(DEVICE), $(filter $(DEVICE),GPU CUDA))
MIN_COMPUTE:=50
ifeq ($(TENSOR), ON)
MIN_COMPUTE:=80
endif
TARGETS_SUPPORTED := $(shell ./test_cuda.sh nvcc "$(GPU_INCLUDE_PATH)" "$(GPU_LIBRARY_PATH)" "$(TARGETS)" "$(MIN_COMPUTE)")
# if user specifies DEVICE=GPU the test result determines wether CUDA will be used or not
ifeq ($(TARGETS_SUPPORTED),)
ifeq ($(DEVICE),CUDA)
$(error Cuda verification failed)
else
$(info Cuda is not available, using OpenCL)
$(info )
override DEVICE:=GPU
export
endif
else
override TARGETS:=$(TARGETS_SUPPORTED)
export
override DEVICE:=CUDA
endif
endif
ifeq ($(DEVICE),CUDA)
override DEVICE:=GPU
export
include Makefile.Cuda
else
ifeq ($(DEVICE),$(filter $(DEVICE),OCLGPU OPENCL))
override DEVICE:=GPU
export
$(info Using OpenCL)
$(info )
endif
$(info Please make sure to set environment variables)
$(info GPU_INCLUDE_PATH and GPU_LIBRARY_PATH)
$(info )
include Makefile.OpenCL
endif

endif # DEVICE=HIP
