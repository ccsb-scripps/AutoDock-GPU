FROM ubuntu:bionic

# Default values for the build
ARG git_branch
ARG git_slug
ARG cxx_compiler
ARG test_ls

ENV lsmet sw
ENV nrun  10

RUN apt-get -yq update

# Utilities
RUN apt-get install -yq --allow-downgrades --allow-remove-essential            \
    --allow-change-held-packages git wget apt-utils cmake unzip                \
    libboost-all-dev software-properties-common 

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

RUN apt-get -yq update

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
RUN bash /AutoDock-GPU/.travis/install_intel_opencl.sh; fi

CMD cd /AutoDock-GPU/ && \
	make DEVICE=CPU NRUN=${nrun} TESTLS=${lsmet} test
