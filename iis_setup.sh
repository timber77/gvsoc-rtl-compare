#!/usr/bin/env bash
export BENDER=bender-0.28.1
# export CC=gcc-9.2.0
# export CXX=g++-9.2.0
export VCS_SEPP=vcs-2020.12
export VERILATOR_SEPP=verilator-5.020
export QUESTA_SEPP=questa-2022.3
export LLVM_BINROOT=/usr/pack/riscv-1.0-kgf/pulp-llvm-0.12.0/bin

export CXX=g++-11.2.0
export CC=gcc-11.2.0
export CMAKE=cmake-3.18.1

source /home/sem24h19/example_floonoc_gvsoc/sourceme_systemc.sh
source /scratch/sem24h19/venv/bin/activate

