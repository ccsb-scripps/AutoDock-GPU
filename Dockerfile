FROM ubuntu:bionic

# Build arguments
ARG git_branch
ARG git_slug

# Execution arguments
ENV lsmet sw
ENV nrun  10
ENV ngen  1000

# Utilities
RUN apt-get -yq update

RUN apt-get install -yq --allow-downgrades --allow-remove-essential            \
    --allow-change-held-packages git wget apt-utils cmake unzip clinfo         \
    g++ gcc clang libboost-all-dev software-properties-common 

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

RUN apt-get -yq update

# OpenCL ICD Loader
RUN apt-get install -yq --allow-downgrades --allow-remove-essential           \
    --allow-change-held-packages ocl-icd-opencl-dev ocl-icd-dev opencl-headers

# AutoDock-GPU
RUN git clone https://github.com/${git_slug}.git -b ${git_branch} /AutoDock-GPU

# Intel OpenCL Runtime
RUN bash /AutoDock-GPU/.travis/install_intel_opencl.sh; fi

CMD clinfo && \
    cd /AutoDock-GPU/ && \
    make DEVICE=CPU TESTLS=${lsmet} NRUN=${nrun} NGEN=${ngen} test && \
    tail -30 test.dlg
