#!/bin/bash

set -ev

###########################
# Get Intel OpenCL Runtime
###########################

# Code taken from here:
# https://github.com/KhronosGroup/SyclParallelSTL/blob/master/.travis/install_intel_opencl.sh
#
# Intel OpenCL drivers available here:
# https://software.intel.com/en-us/articles/opencl-drivers#latest_CPU_runtime

PACKAGE_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/12556/opencl_runtime_16.1.2_x64_rh_6.4.0.37.tgz
PACKAGE_NAME=opencl_runtime_16.1.2_x64_rh_6.4.0.37

wget -q ${PACKAGE_URL} -O /tmp/opencl_runtime.tgz
tar -xzf /tmp/opencl_runtime.tgz -C /tmp
sed 's/decline/accept/g' -i /tmp/${PACKAGE_NAME}/silent.cfg
sudo apt-get install -yq cpio
/tmp/${PACKAGE_NAME}/install.sh -s /tmp/${PACKAGE_NAME}/silent.cfg
