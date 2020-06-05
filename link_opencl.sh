#!/bin/bash
# Link OpenCL code files for compilation

if [[ -f host/src/setup.cpp ]] || [[ ! -f host/src/calcenergy.cpp ]]; then
	ln -sf calcenergy.h.OpenCL host/inc/calcenergy.h
	ln -sf getparameters.h.OpenCL host/inc/getparameters.h
	ln -sf performdocking.h.OpenCL host/inc/performdocking.h
	ln -sf processligand.h.OpenCL host/inc/processligand.h
	ln -sf miscellaneous.h.OpenCL host/inc/miscellaneous.h
	ln -sf processgrid.h.OpenCL host/inc/processgrid.h
	ln -sf processresult.h.OpenCL host/inc/processresult.h
	ln -sf calcenergy.cpp.OpenCL host/src/calcenergy.cpp
	ln -sf main.cpp.OpenCL host/src/main.cpp
	ln -sf performdocking.cpp.OpenCL host/src/performdocking.cpp
	ln -sf processligand.cpp.OpenCL host/src/processligand.cpp
	ln -sf getparameters.cpp.OpenCL host/src/getparameters.cpp
	ln -sf miscellaneous.cpp.OpenCL host/src/miscellaneous.cpp
	ln -sf processgrid.cpp.OpenCL host/src/processgrid.cpp
	ln -sf processresult.cpp.OpenCL host/src/processresult.cpp
	rm -f host/src/setup.cpp
fi
