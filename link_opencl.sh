#!/bin/bash
# Link OpenCL code files for compilation

if [[ -f host/src/setup.cpp ]] || [[ ! -f host/src/performdocking.cpp ]]; then
	ln -sf performdocking.h.OpenCL host/inc/performdocking.h
	ln -sf main.cpp.OpenCL host/src/main.cpp
	ln -sf performdocking.cpp.OpenCL host/src/performdocking.cpp
fi
