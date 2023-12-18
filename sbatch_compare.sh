#!/bin/bash -l
#SBATCH --job-name=compare
#SBATCH --output=compare.out
#SBATCH --error=compare.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1

# CXX=clang++ CC=clang BUILD_DIR=clang-build IGNORE_AUTOTUNED=1 ./compare.sh
CXX=g++ CC=gcc BUILD_DIR=gcc-build ./compare.sh
