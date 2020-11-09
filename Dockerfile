FROM ubuntu:bionic

# Default values for the build
ARG git_branch
ARG git_slug
ARG cxx_compiler
ARG target

RUN apt-get -yq update

# Utilities
RUN apt-get install -yq --allow-downgrades --allow-remove-essential            \
    --allow-change-held-packages git wget apt-utils cmake unzip                \
    libboost-all-dev software-properties-common 

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

RUN apt-get -yq update

# Clang 6.0
RUN if [ "${c_compiler}" = 'clang-6.0' ]; then apt-get install -yq             \
    --allow-downgrades --allow-remove-essential --allow-change-held-packages   \
     clang-6.0 libomp-dev; fi

# GCC 7
RUN if [ "${c_compiler}" = 'gcc-7' ]; then apt-get install -yq                 \
    --allow-downgrades --allow-remove-essential --allow-change-held-packages   \
    g++-7 gcc-7; fi

# OpenCL ICD Loader
RUN apt-get install -yq --allow-downgrades --allow-remove-essential           \
    --allow-change-held-packages ocl-icd-opencl-dev ocl-icd-dev opencl-headers

# AutoDock-GPU
RUN git clone https://github.com/${git_slug}.git -b ${git_branch} /AutoDock-GPU

# Intel OpenCL Runtime
RUN if [ "${target}" = 'opencl' ]; then bash /AutoDock-GPU/.travis/install_intel_opencl.sh; fi

#CMD
