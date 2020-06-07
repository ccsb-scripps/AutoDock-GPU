#!/bin/bash
# Link Cuda code files for compilation

if [[ ! -f host/src/setup.cpp ]]; then
	ln -sf performdocking.h.Cuda host/inc/performdocking.h
	ln -sf main.cpp.Cuda host/src/main.cpp
	ln -sf performdocking.cpp.Cuda host/src/performdocking.cpp
fi
