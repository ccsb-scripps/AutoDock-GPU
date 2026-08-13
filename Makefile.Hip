# AutoDock-GPU HIP Makefile (ROCm/AMD backend, DEVICE=HIP)
#
# Mirrors Makefile.Cuda but compiles the shared cuda/ kernel translation unit
# with hipcc and links the HIP runtime. The host .cpp are compiled by g++ with
# the HIP runtime headers (host-side HIP API only). Selected via DEVICE=HIP in
# the top-level Makefile.
#
# Arch is configurable and never hardcoded, so any GPU target builds
# with only HIP_ARCH=<arch> and no source/Makefile edit:
#   make DEVICE=HIP NUMWI=64 HIP_ARCH=gfx90a

ROCM_PATH ?= /opt/rocm
HIPCC = $(ROCM_PATH)/bin/hipcc
CPP = g++
UNAME := $(shell uname)

# Default the target GPU arch ONLY when unset; a passed HIP_ARCH must win.
HIP_ARCH ?= gfx90a

# TENSOR=ON accelerates the energy/gradient sum-reduction with matrix-core units.
# On AMD this is rocWMMA over native fp32 MFMA (header-only, no extra link); it
# activates the same USE_NVTENSOR call sites the CUDA backend uses. The reduction
# fills a wavefront-wide collective, so it requires NUMWI > 32 (as on CUDA).
ifeq ($(TENSOR), ON)
NVTENSOR = -DUSE_NVTENSOR
ifeq ($(filter 64 128 256,$(NUMWI)),)
$(error TENSOR=ON requires NUMWI greater than 32 (use NUMWI=64, 128, or 256))
endif
endif

ifeq ($(DEVICE), CPU)
	DEV =-DCPU_DEVICE
else ifeq ($(DEVICE), GPU)
	DEV =-DGPU_DEVICE
endif

# ------------------------------------------------------
# Project directories
COMMON_DIR=./common
HOST_INC_DIR=./host/inc
HOST_SRC_DIR=./host/src
KRNL_DIR=./cuda
KCMN_DIR=$(COMMON_DIR)
BIN_DIR=./bin

TARGET := autodock
TOOL_TARGET := adgpu_analysis

# HIP runtime headers/libs for both the hipcc kernel TU and the g++ host TUs.
HIP_INCLUDES = -I$(ROCM_PATH)/include
HIP_HOST_FLAGS = -DUSE_HIP -D__HIP_PLATFORM_AMD__ $(HIP_INCLUDES)
LIB_HIP = kernels.o -L$(ROCM_PATH)/lib -lamdhip64 -DUSE_HIP

IFLAGS=-I$(COMMON_DIR) -I$(HOST_INC_DIR) -I$(KRNL_DIR) $(HIP_INCLUDES)
LFLAGS=-Wl,-rpath=$(ROCM_PATH)/lib
CFLAGS=-std=c++17 $(IFLAGS) $(LFLAGS)
TOOL_CFLAGS=-std=c++17 -I$(COMMON_DIR) -I$(HOST_INC_DIR)

ifeq ($(DEVICE), CPU)
	TARGET:=$(TARGET)_cpu
else ifeq ($(DEVICE), GPU)
	NWI=-DN64WI
	TARGET:=$(TARGET)_gpu
endif

ifeq ($(OVERLAP), ON)
	PIPELINE=-DUSE_PIPELINE -fopenmp
endif

BIN := $(wildcard $(TARGET)*)

# ------------------------------------------------------
# Number of work-items (wi)
# Valid values: 32, 64, 128, 256
NUMWI=

ifeq ($(NUMWI), 32)
	NWI=-DN32WI
	TARGET:=$(TARGET)_32wi
else ifeq ($(NUMWI), 64)
	NWI=-DN64WI
	TARGET:=$(TARGET)_64wi
else ifeq ($(NUMWI), 128)
	NWI=-DN128WI
	TARGET:=$(TARGET)_128wi
else ifeq ($(NUMWI), 256)
		NWI=-DN256WI
		TARGET:=$(TARGET)_256wi
else
	ifeq ($(DEVICE), CPU)
		NWI=-DN16WI
		TARGET:=$(TARGET)_16wi
	else ifeq ($(DEVICE), GPU)
		NWI=-DN128WI
		TARGET:=$(TARGET)_128wi
	endif
endif

# ------------------------------------------------------
# Configuration
# FDEBUG (full) : enables debugging on both host + device
# LDEBUG (light): enables debugging on host
# RELEASE
CONFIG=RELEASE
HIP_FLAGS = --offload-arch=$(HIP_ARCH) -std=c++17

ifeq ($(CONFIG),FDEBUG)
	OPT =-O0 -g3 -Wall -DDOCK_DEBUG
	HIP_FLAGS += -g -ffast-math
else ifeq ($(CONFIG),LDEBUG)
	OPT =-O0 -g3 -Wall
	HIP_FLAGS += -ffast-math
else ifeq ($(CONFIG),RELEASE)
	OPT =-O3
	HIP_FLAGS += -O3 -ffast-math
else
	OPT =
	HIP_FLAGS += -ffast-math
endif

# ------------------------------------------------------
# Reproduce results (remove randomness)
REPRO=NO

ifeq ($(REPRO),YES)
	REP =-DREPRO
else
	REP =
endif
# ------------------------------------------------------

all: otool odock

check-env-dev:
	@if test -z "$$DEVICE"; then \
		echo "Please set DEVICE to either CPU, GPU, CUDA, OCLGPU, or HIP to build docking software."; \
		exit 1; \
	else \
		if [ "$$DEVICE" = "GPU" ]; then \
			echo "DEVICE is set to $$DEVICE"; \
		else \
			echo "DEVICE value is invalid. Please set DEVICE to either CPU, GPU, CUDA, OCLGPU, or HIP"; \
			exit 1; \
		fi; \
	fi; \
	echo " "

check-env-all: check-env-dev

# ------------------------------------------------------
# Printing out its git version hash

GIT_VERSION := $(shell ./version_string.sh)

CFLAGS+=-DVERSION=\"$(GIT_VERSION)\"
TOOL_CFLAGS+=-DVERSION=\"$(GIT_VERSION)\"

# ------------------------------------------------------

kernels: $(KERNEL_SRC)
	$(HIPCC) $(NWI) $(NVTENSOR) $(REP) $(HIP_FLAGS) -fPIE -DUSE_HIP $(IFLAGS) -c $(KRNL_DIR)/kernels.cu -o kernels.o

otool:
	@echo "Building" $(TOOL_TARGET) "..."
	$(CPP) \
	$(shell ls $(HOST_SRC_DIR)/*.cpp) \
	$(TOOL_CFLAGS) \
	-o$(BIN_DIR)/$(TOOL_TARGET) \
	$(PIPELINE) $(OPT) -DTOOLMODE $(REP)

odock: check-env-all kernels
	@echo "Building" $(TARGET) "..."
	$(CPP) \
	$(shell ls $(HOST_SRC_DIR)/*.cpp) \
	$(CFLAGS) \
	$(HIP_HOST_FLAGS) \
	$(LIB_HIP) \
	-o$(BIN_DIR)/$(TARGET) \
	$(DEV) $(NWI) $(NVTENSOR) $(PIPELINE) $(OPT) $(DD) $(REP) $(KFLAGS)

PDB      := 3ce3
NRUN     := 100
NGEN     := 27000
POPSIZE  := 150
TESTNAME := test
TESTLS   := ad

test: odock
	$(BIN_DIR)/$(TARGET) \
	-ffile ./input/$(PDB)/derived/$(PDB)_protein.maps.fld \
	-lfile ./input/$(PDB)/derived/$(PDB)_ligand.pdbqt \
	-nrun $(NRUN) \
	-ngen $(NGEN) \
	-psize $(POPSIZE) \
	-resnam $(TESTNAME) \
	-gfpop 0 \
	-lsmet $(TESTLS)

.PHONY: clean
clean:
	rm -f kernels.o $(BIN_DIR)/$(TARGET)* $(BIN_DIR)/$(TOOL_TARGET)*
