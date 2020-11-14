#!/bin/bash

set -o xtrace

###########################
# Run test
###########################

# source ./run_test.sh

info="[INFO]"
asterix_line="========================================================================"

function verify_sourced_script() {
  echo " "
  echo "${lines}"
  echo "${info}: Verifying if script is being sourced"
  echo "${lines}"
  echo " "
  if [[ ${BASH_SOURCE[0]} != $0 ]]; then
    echo "${info}: OK. Script "${BASH_SOURCE[0]}" is sourced"
  else
    echo "${info}: Run \"source ${BASH_SOURCE[0]}\" instead!"
    echo "Terminating!" && exit 1
  fi
}

function run_clinfo() {
    echo " "
    echo "${asterix_line}"
    echo "${info}: Running clinfo"
    echo "${asterix_line}"
    echo " "
    clinfo
}

#ADGPU_DIR=~/Desktop/AutoDock-GPU
ADGPU_DIR=/AutoDock-GPU
BIN_DIR=${ADGPU_DIR}/bin
DEVICE=CPU

function compile_adgpu() {
    echo " "
    echo "${asterix_line}"
    echo "${info}: Compiling AutoDock-GPU"
    echo "${asterix_line}"
    echo " "
    cd ${ADGPU_DIR}
    make DEVICE=${DEVICE} NUMWI=${1}
    ls -asl ${BIN_DIR}
}

function run_test() {
    echo " "
    echo "${asterix_line}"
    echo "${info}: Running automated test"
    echo "${asterix_line}"
    echo " "
    for pdb in "3ce3" "1stp"; do
        for lsmet in "sw" "ad"; do
            for nrun in 1 10; do
                for ngen in 10 100; do
                    for psize in 10 50; do
                        ${BIN_DIR}/autodock_cpu_${1}wi \
                        -ffile ./input/${pdb}/derived/${pdb}_protein.maps.fld \
                        -lfile ./input/${pdb}/derived/${pdb}_ligand.pdbqt \
                        -lsmet ${lsmet} \
                        -nrun ${nrun} \
                        -ngen ${ngen} \
                        -psize ${psize} \
                        -resnam ${pdb}-${lsmet}-${nrun}-${ngen}-${psize}-"`date +"%Y-%m-%d-%H:%M"`" \
                        -xmloutput 0
                        echo " "
	                    tail -30 ${pdb}-${lsmet}-${nrun}-${ngen}-${psize}-*.dlg
                    done
                done
            done
        done
    done        
    echo " "
    ls -asl *.dlg
}

verify_sourced_script
run_clinfo
compile_adgpu ${1}
run_test ${1}