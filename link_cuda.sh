#!/bin/bash
# Link Cuda code files for compilation

if [[ ! -f host/src/setup.cpp ]]; then
	ln -sf calcenergy.h.Cuda host/inc/calcenergy.h
	ln -sf getparameters.h.Cuda host/inc/getparameters.h
	ln -sf performdocking.h.Cuda host/inc/performdocking.h
	ln -sf processligand.h.Cuda host/inc/processligand.h
	ln -sf miscellaneous.h.Cuda host/inc/miscellaneous.h
	ln -sf processgrid.h.Cuda host/inc/processgrid.h
	ln -sf processresult.h.Cuda host/inc/processresult.h
	ln -sf calcenergy.cpp.Cuda host/src/calcenergy.cpp
	ln -sf main.cpp.Cuda host/src/main.cpp
	ln -sf performdocking.cpp.Cuda host/src/performdocking.cpp
	ln -sf processligand.cpp.Cuda host/src/processligand.cpp
	ln -sf getparameters.cpp.Cuda host/src/getparameters.cpp
	ln -sf miscellaneous.cpp.Cuda host/src/miscellaneous.cpp
	ln -sf processgrid.cpp.Cuda host/src/processgrid.cpp
	ln -sf processresult.cpp.Cuda host/src/processresult.cpp
	ln -sf setup.cpp.Cuda host/src/setup.cpp
fi
