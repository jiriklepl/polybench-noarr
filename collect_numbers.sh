#!/bin/bash

BUILD_DIR=${BUILD_DIR:-build}
DATASET_SIZE=${DATASET_SIZE:-EXTRALARGE}
DATA_TYPE=${DATA_TYPE:-FLOAT}
NOARR_STRUCTURES_BRANCH=${NOARR_STRUCTURES_BRANCH:-tuning}

if [ -z "$POLYBENCH_C_DIR" ]; then
	POLYBENCH_C_DIR="$BUILD_DIR/PolyBenchC-4.2.1"
	mkdir -p "$POLYBENCH_C_DIR" || exit 1
	if [ -d "$POLYBENCH_C_DIR/.git" ]; then
		( cd "$POLYBENCH_C_DIR" && git pull )
	else
		git clone "https://github.com/jiriklepl/PolyBenchC-4.2.1.git" "$POLYBENCH_C_DIR"
	fi
fi

( cd "$POLYBENCH_C_DIR" && ./build.sh ) || exit 1
( cd . && ./build.sh ) || exit 1


find build -maxdepth 1 -executable -type f \
	| while read -r file; do

    filename=$(basename "$file")

    srun -A kdss -p mpi-homo-short --exclusive -ww201 \
        ./run_algorithm.sh "$file" > "$filename"

    done
