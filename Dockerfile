FROM ubuntu:bionic

# Default values for the build
ARG git_branch
ARG git_slug
ARG target

#SHELL [ "/bin/bash" ]

# Install utilities
RUN apt-get -yq update

RUN apt-get install -yq git wget apt-utils cmake unzip gcc g++ clang

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

RUN apt-get -yq update


# AutoDock-GPU
RUN git clone https://github.com/${git_slug}.git -b ${git_branch} /AutoDock-GPU

# Intel OpenCL Runtime
RUN if [ "${target}" = 'opencl' ]; then bash /AutoDock-GPU/.travis/install_intel_opencl.sh; fi

