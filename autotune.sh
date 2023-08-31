#!/bin/bash -ex

mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_CXX_FLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -D_POSIX_C_SOURCE=200809L"
cmake --build . -j$(nproc)
python <(./gemm_autotune) --test-limit=100 --no-dups
