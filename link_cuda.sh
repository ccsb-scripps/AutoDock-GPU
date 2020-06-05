#!/bin/bash
# Link Cuda code files for compilation

if [[ ! -f host/src/setup.cpp ]]; then
	ln -sf getparameters.h.Cuda host/inc/getparameters.h
	ln -sf performdocking.h.Cuda host/inc/performdocking.h
	ln -sf miscellaneous.h.Cuda host/inc/miscellaneous.h
	ln -sf processresult.h.Cuda host/inc/processresult.h
	ln -sf main.cpp.Cuda host/src/main.cpp
	ln -sf performdocking.cpp.Cuda host/src/performdocking.cpp
	ln -sf getparameters.cpp.Cuda host/src/getparameters.cpp
	ln -sf miscellaneous.cpp.Cuda host/src/miscellaneous.cpp
	ln -sf processresult.cpp.Cuda host/src/processresult.cpp
	ln -sf setup.cpp.Cuda host/src/setup.cpp
fi
