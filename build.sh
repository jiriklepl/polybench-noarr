#!/bin/bash -ex

# Create the build directory
cmake -E make_directory build
cd build

# Configure the build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE " \
    ..

# Build the project
NPROC=$(nproc)
cmake --build . --config Release -j"$NPROC"
