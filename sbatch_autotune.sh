#!/bin/bash -l
#SBATCH --job-name=autotune
#SBATCH --output=autotune.out
#SBATCH --error=autotune.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1

# CXX=clang++ CC=clang BUILD_DIR=clang-build MAX_RUNS=100 ./autotune.sh
CXX=g++ CC=gcc BUILD_DIR=gcc-build MAX_RUNS=100 ./autotune.sh
