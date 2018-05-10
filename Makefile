# OCLADock Makefile

# ------------------------------------------------------
# Note that environment variables must be defined
# before compiling
# DEVICE?
# if DEVICE=CPU: CPU_INCLUDE_PATH?, CPU_LIBRARY_PATH?
# if DEVICE=GPU: GPU_INCLUDE_PATH?, GPU_LIBRARY_PATH?

# ------------------------------------------------------
# Choose OpenCL device
# Valid values: CPU, GPU

ifeq ($(DEVICE), CPU)
	DEV =-DCPU_DEVICE
	OCLA_INC_PATH=$(CPU_INCLUDE_PATH)
	OCLA_LIB_PATH=$(CPU_LIBRARY_PATH)
else ifeq ($(DEVICE), GPU)
	DEV =-DGPU_DEVICE
	OCLA_INC_PATH=$(GPU_INCLUDE_PATH)
	OCLA_LIB_PATH=$(GPU_LIBRARY_PATH)
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
K5_NAME="gradient_minimizer"
K_NAMES=-DK1=$(K1_NAME) -DK2=$(K2_NAME) -DK3=$(K3_NAME) -DK4=$(K4_NAME) -DK5=$(K5_NAME)
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
else ifeq ($(NUMWI), 256)
		NWI=-DN256WI
		TARGET:=$(TARGET)_256wi
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

OCL_DEBUG_BASIC=-DPLATFORM_ATTRIBUTES_DISPLAY\
	        -DDEVICE_ATTRIBUTES_DISPLAY

OCL_DEBUG_ALL=$(OCL_DEBUG_BASIC) \
	      -DCONTEXT_INFO_DISPLAY \
	      -DCMD_QUEUE_INFO_DISPLAY \
	      -DCMD_QUEUE_PROFILING_ENABLE \
	      -DCMD_QUEUE_OUTORDER_ENABLE \
	      -DPROGRAM_INFO_DISPLAY \
	      -DPROGRAM_BUILD_INFO_DISPLAY \
	      -DKERNEL_INFO_DISPLAY \
	      -DKERNEL_WORK_GROUP_INFO_DISPLAY \
	      -DBUFFER_OBJECT_INFO_DISPLAY

ifeq ($(CONFIG),DEBUG)
	OPT =-O0 -g3 -Wall $(OCL_DEBUG_BASIC)
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

check-env-dev:
	@if test -z "$$DEVICE"; then \
		echo "DEVICE is undefined"; \
		exit 1; \
	else \
		if [ "$$DEVICE" = "CPU" ]; then \
			echo "DEVICE is set to $$DEVICE"; \
		else \
			if [ "$$DEVICE" = "GPU" ]; then \
				echo "DEVICE is set to $$DEVICE"; \
			else \
				echo "DEVICE value is invalid. Set DEVICE to either CPU or GPU"; \
			fi; \
		fi; \
	fi; \
	echo " "

check-env-cpu:
	@if test -z "$$CPU_INCLUDE_PATH"; then \
		echo "CPU_INCLUDE_PATH is undefined"; \
	else \
		echo "CPU_INCLUDE_PATH is set to $$CPU_INCLUDE_PATH"; \
	fi; \
	if test -z "$$CPU_LIBRARY_PATH"; then \
		echo "CPU_LIBRARY_PATH is undefined"; \
	else \
		echo "CPU_LIBRARY_PATH is set to $$CPU_LIBRARY_PATH"; \
	fi; \
	echo " "

check-env-gpu:
	@if test -z "$$GPU_INCLUDE_PATH"; then \
		echo "GPU_INCLUDE_PATH is undefined"; \
	else \
		echo "GPU_INCLUDE_PATH is set to $$GPU_INCLUDE_PATH"; \
	fi; \
	if test -z "$$GPU_LIBRARY_PATH"; then \
		echo "GPU_LIBRARY_PATH is undefined"; \
	else \
		echo "GPU_LIBRARY_PATH is set to $$GPU_LIBRARY_PATH"; \
	fi; \
	echo " "

check-env-all: check-env-dev check-env-cpu check-env-gpu

stringify:
	./stringify_ocl_krnls.sh

odock: check-env-all stringify $(SRC)
	g++ $(SRC) $(CFLAGS) -lOpenCL -o$(BIN_DIR)/$(TARGET) $(DEV) $(NWI) $(OPT) $(DD) $(REP) $(KFLAGS)


# Example
# 1ac8: for testing gradients of translation and rotation genes
# 7cpa: for testing gradients of torsion genes (15 torsions) 
# 3tmn: for testing gradients of torsion genes (1 torsion)

PDB     := 3ce3
NRUN    := 100
POPSIZE := 150
TESTNAME:= test
TESTLS  := sw

test: odock
	$(BIN_DIR)/$(TARGET) -ffile ./input/$(PDB)/derived/$(PDB)_protein.maps.fld -lfile ./input/$(PDB)/derived/$(PDB)_ligand.pdbqt -nrun $(NRUN) -psize $(POPSIZE) -resnam $(TESTNAME) -gfpop 1 -lsmet $(TESTLS)

ASTEX_PDB := 2bsm
ASTEX_NRUN:= 10
ASTEX_POPSIZE := 10
ASTEX_TESTNAME := test_astex
ASTEX_LS := sw

astex: odock
	$(BIN_DIR)/$(TARGET) -ffile ./input_tsri/search-set-astex/$(ASTEX_PDB)/protein.maps.fld -lfile ./input_tsri/search-set-astex/$(ASTEX_PDB)/flex-xray.pdbqt -nrun $(ASTEX_NRUN) -psize $(ASTEX_POPSIZE) -resnam $(ASTEX_TESTNAME) -gfpop 1 -lsmet $(ASTEX_LS)


#	$(BIN_DIR)/$(TARGET) -ffile ./input_tsri/search-set-astex/$(ASTEX_PDB)/protein.maps.fld -lfile ./input_tsri/search-set-astex/$(ASTEX_PDB)/flex-xray.pdbqt -nrun $(ASTEX_NRUN) -psize $(ASTEX_POPSIZE) -resnam $(ASTEX_TESTNAME) -gfpop 1 | tee ./input_tsri/search-set-astex/intrapairs/$(ASTEX_PDB)_intrapair.txt

clean:
	rm -f $(BIN_DIR)/* initpop.txt
