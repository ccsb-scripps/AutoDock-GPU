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

CPP = g++ -std=c++11 -stdlib=libc++

UNAME := $(shell uname)


KOKKOS_INC_PATH=/Users/ascheinberg/Research/kokkos/install/include/
KOKKOS_LIB_PATH=/Users/ascheinberg/Research/kokkos/install/lib/
LIB_KOKKOS=-lkokkoscore

# ------------------------------------------------------
# Project directories
# opencl_lvs: wrapper for OpenCL APIs
COMMON_DIR=./common
HOST_INC_DIR=./host/inc
HOST_SRC_DIR=./host/src
KCODE_INC_PATH=./kokkos
BIN_DIR=./bin

# Host sources
HOST_SRC=$(wildcard $(HOST_SRC_DIR)/*.cpp)
SRC=$(OCL_SRC) $(HOST_SRC)

IFLAGS=-I$(COMMON_DIR) -I$(HOST_INC_DIR) -I$(KOKKOS_INC_PATH) -I$(KCODE_INC_PATH)
LFLAGS=-L$(KOKKOS_LIB_PATH) $(LIB_KOKKOS)
CFLAGS=$(IFLAGS) $(LFLAGS)

TARGET := autodock
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

ifeq ($(NUMWI), 1)
	NWI=-DN1WI
	TARGET:=$(TARGET)_1wi
else ifeq ($(NUMWI), 2)
	NWI=-DN2WI
	TARGET:=$(TARGET)_2wi
else ifeq ($(NUMWI), 4)
	NWI=-DN4WI
	TARGET:=$(TARGET)_4wi
else ifeq ($(NUMWI), 8)
	NWI=-DN8WI
	TARGET:=$(TARGET)_8wi
else ifeq ($(NUMWI), 16)
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
# Configuration
# FDEBUG (full) : enables debugging on both host + device
# LDEBUG (light): enables debugging on host
# RELEASE
CONFIG=RELEASE
#CONFIG=FDEBUG

OCL_DEBUG_BASIC=-DPLATFORM_ATTRIBUTES_DISPLAY\
	      -DCMD_QUEUE_PROFILING_ENABLE \
	        -DDEVICE_ATTRIBUTES_DISPLAY

OCL_DEBUG_ALL=$(OCL_DEBUG_BASIC) \
	      -DCONTEXT_INFO_DISPLAY \
	      -DCMD_QUEUE_INFO_DISPLAY \
	      -DCMD_QUEUE_PROFILING_ENABLE \
	      -DPROGRAM_INFO_DISPLAY \
	      -DPROGRAM_BUILD_INFO_DISPLAY \
	      -DKERNEL_INFO_DISPLAY \
	      -DKERNEL_WORK_GROUP_INFO_DISPLAY \
	      -DBUFFER_OBJECT_INFO_DISPLAY
ifneq ($(UNAME), Darwin) # out of order queues don't work on Mac OS X
OCL_DEBUG_ALL += -DCMD_QUEUE_OUTORDER_ENABLE
endif

ifeq ($(CONFIG),FDEBUG)
	OPT =-O0 -g3 -Wall $(OCL_DEBUG_ALL) -DDOCK_DEBUG
else ifeq ($(CONFIG),LDEBUG)
	OPT =-O0 -g3 -Wall $(OCL_DEBUG_BASIC)
else ifeq ($(CONFIG),RELEASE)
	OPT =-O3
else
	OPT =
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

# ------------------------------------------------------
# Priting out its git version hash

GIT_VERSION := $(shell git describe --abbrev=40 --dirty --always --tags)

CFLAGS+=-DVERSION=\"$(GIT_VERSION)\"

# ------------------------------------------------------

odock: check-env-all $(SRC)
	$(CPP) \
	$(SRC) \
	$(CFLAGS) \
	-o$(BIN_DIR)/$(TARGET) \
	$(NWI) $(OPT) $(DD) $(REP)

# Example
# 1ac8: for testing gradients of translation and rotation genes
# 7cpa: for testing gradients of torsion genes (15 torsions) 
# 3tmn: for testing gradients of torsion genes (1 torsion)

PDB      := 3ce3
NRUN     := 100
NGEN     := 27000
POPSIZE  := 150
TESTNAME := test
TESTLS   := sw

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

ASTEX_PDB := 2bsm
ASTEX_NRUN:= 10
ASTEX_POPSIZE := 10
ASTEX_TESTNAME := test_astex
ASTEX_LS := sw

astex: odock
	$(BIN_DIR)/$(TARGET) \
	-ffile ./input_tsri/search-set-astex/$(ASTEX_PDB)/protein.maps.fld \
	-lfile ./input_tsri/search-set-astex/$(ASTEX_PDB)/flex-xray.pdbqt \
	-nrun $(ASTEX_NRUN) \
	-psize $(ASTEX_POPSIZE) \
	-resnam $(ASTEX_TESTNAME) \
	-gfpop 1 \
	-lsmet $(ASTEX_LS)

#	$(BIN_DIR)/$(TARGET) -ffile ./input_tsri/search-set-astex/$(ASTEX_PDB)/protein.maps.fld -lfile ./input_tsri/search-set-astex/$(ASTEX_PDB)/flex-xray.pdbqt -nrun $(ASTEX_NRUN) -psize $(ASTEX_POPSIZE) -resnam $(ASTEX_TESTNAME) -gfpop 1 | tee ./input_tsri/search-set-astex/intrapairs/$(ASTEX_PDB)_intrapair.txt

clean:
	rm -f $(BIN_DIR)/* initpop.txt
