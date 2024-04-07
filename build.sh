#!/bin/bash -ex

# Create the build directory
cmake -E make_directory build
cd build

export DATASET_SIZE=${DATASET_SIZE:-EXTRALARGE}
export DATA_TYPE=${DATA_TYPE:-FLOAT}
export NOARR_STRUCTURES_BRANCH=${NOARR_STRUCTURES_BRANCH:-main}

# Configure the build
cmake -DCMAKE_BUILD_TYPE=Release \
    -DNOARR_STRUCTURES_BRANCH="$NOARR_STRUCTURES_BRANCH" \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -D${DATASET_SIZE}_DATASET -DDATA_TYPE_IS_$DATA_TYPE" \
    ..

# Build the project
NPROC=$(nproc)
cmake --build . --config Release -j"$NPROC"
