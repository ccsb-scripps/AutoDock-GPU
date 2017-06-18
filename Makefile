# ocladock Makefile

# CPU config
INTEL_INCLUDE_PATH=$(INTELOCLSDKROOT)/include
INTEL_LIBRARY_PATH=$(INTELOCLSDKROOT)

# GPU config
AMD_INCLUDE_PATH=$(AMDAPPSDKROOT)/include
AMD_LIBRARY_PATH=$(shell cat /etc/ld.so.conf.d/amdgpu-pro-x86_64.conf)

# ------------------------------------------------------
# Choose OpenCL device
# Valid values: CPU, GPU
DEVICE=GPU

ifeq ($(DEVICE), CPU)
	DEV =-DCPU_DEVICE
	OCLA_INC_PATH=$(INTEL_INCLUDE_PATH)
	OCLA_LIB_PATH=$(INTEL_LIBRARY_PATH)
else ifeq ($(DEVICE), GPU)
	DEV =-DGPU_DEVICE
	OCLA_INC_PATH=$(AMD_INCLUDE_PATH)
	OCLA_LIB_PATH=$(AMD_LIBRARY_PATH)
endif

# ------------------------------------------------------
# Project directories
# opencl_lvs: wrapper for OpenCL APIs
COMMON_DIR=./common
OCL_INC_DIR=./wrapcl/inc
OCL_SRC_DIR=./wrapcl/src
HOST_INC_DIR=./host/inc
HOST_SRC_DIR=./host/src
KRNL_DIR=./device
KCMN_DIR=$(COMMON_DIR)
BIN_DIR=./bin

# Host sources
OCL_SRC=$(wildcard $(OCL_SRC_DIR)/*.cpp)
HOST_SRC=$(wildcard $(HOST_SRC_DIR)/*.cpp)
SRC=$(OCL_SRC) $(HOST_SRC)

IFLAGS=-I$(COMMON_DIR) -I$(OCL_INC_DIR) -I$(HOST_INC_DIR) -I$(OCLA_INC_PATH)
LFLAGS=-L$(OCLA_LIB_PATH)
CFLAGS=$(IFLAGS) $(LFLAGS)

# Device sources
KRNL_MAIN=calcenergy.cl
KRNL_SRC=$(KRNL_DIR)/$(KRNL_MAIN)
# Kernel names
K1_NAME="gpu_calc_initpop"
K2_NAME="gpu_sum_evals"
K3_NAME="perform_LS"
K4_NAME="gpu_gen_and_eval_newpops"
K_NAMES=-DK1=$(K1_NAME) -DK2=$(K2_NAME) -DK3=$(K3_NAME) -DK4=$(K4_NAME)
# Kernel flags
KFLAGS=-DKRNL_SOURCE=$(KRNL_DIR)/$(KRNL_MAIN) -DKRNL_DIRECTORY=$(KRNL_DIR) -DKCMN_DIRECTORY=$(KCMN_DIR) $(K_NAMES)

TARGET := ocladock
ifeq ($(DEVICE), CPU)
	TARGET:=$(TARGET)_cpu
else ifeq ($(DEVICE), GPU)
	NWI=-DN64WI
	TARGET:=$(TARGET)_gpu
endif

BIN := $(wildcard $(TARGET)*)

# ------------------------------------------------------
# Number of work-items (wi)
# Valid values: 16, 32, 64, 128
NUMWI=

ifeq ($(NUMWI), 16)
	NWI=-DN16WI
	TARGET:=$(TARGET)_16wi
else ifeq ($(NUMWI), 32)
	NWI=-DN32WI
	TARGET:=$(TARGET)_32wi
else ifeq ($(NUMWI), 64)
	NWI=-DN64WI
	TARGET:=$(TARGET)_64wi
else ifeq ($(NUMWI), 128)
	NWI=-DN128WI
	TARGET:=$(TARGET)_128wi
else
	ifeq ($(DEVICE), CPU)
		NWI=-DN16WI
		TARGET:=$(TARGET)_16wi
	else ifeq ($(DEVICE), GPU)
		NWI=-DN64WI
		TARGET:=$(TARGET)_64wi
	endif
endif

# ------------------------------------------------------
# Configuration (Host)
# Valid values: RELEASE, DEBUG
CONFIG=RELEASE

ifeq ($(CONFIG),DEBUG)
	OPT =-O0 -g3 -Wall
else ifeq ($(CONFIG),RELEASE)
	OPT =-O3
else
	OPT =
endif

# ------------------------------------------------------
# Host and Device Debug
DOCK_DEBUG=NO

# Reproduce results (remove randomness)
REPRO=NO

ifeq ($(DOCK_DEBUG),YES)
	DD =-DDOCK_DEBUG
else
	DD =
endif

ifeq ($(REPRO),YES)
	REP =-DREPRO
else
	REP =
endif
# ------------------------------------------------------

all: odock

odock: $(SRC)
	g++ $(SRC) $(CFLAGS) -lOpenCL -o$(BIN_DIR)/$(TARGET) $(DEV) $(NWI) $(OPT) $(DD) $(REP) $(KFLAGS)

clean:
	rm -f $(BIN_DIR)/* initpop.txt
