#!/bin/bash
# Copyright (C) 2022 Intel Corporation

# Link Cuda code files for compilation

ln -sf performdocking.h.dpcpp host/inc/performdocking.h
ln -sf performdocking.cpp.dpcpp host/src/performdocking.cpp
