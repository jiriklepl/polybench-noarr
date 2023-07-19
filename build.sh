#!/bin/bash

# Create the build directory
cmake -E make_directory build || exit 1
cd build || exit 1

# Configure the build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -D_POSIX_C_SOURCE=200809L " \
    .. \
    || exit 1

# Build the project
NPROC=$(nproc)
cmake --build . --config Release -j"$NPROC" || exit 1
