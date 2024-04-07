#!/bin/bash

export BUILD_DIR=${BUILD_DIR:-build}
export DATASET_SIZE=${DATASET_SIZE:-EXTRALARGE}
export DATA_TYPE=${DATA_TYPE:-FLOAT}
export NOARR_STRUCTURES_BRANCH=${NOARR_STRUCTURES_BRANCH:-main}
export USE_SLURM=${USE_SLURM:-0}
export DATA_DIR=${DATA_DIR:-data}

# SLURM settings (if used)
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-kdss}
export SLURM_PARTITION=${SLURM_PARTITION:-mpi-homo-short}
export SLURM_WORKER=${SLURM_WORKER:-w201}
export SLURM_TIMEOUT=${SLURM_TIMEOUT:-"2:00:00"}

if [ -z "$POLYBENCH_C_DIR" ]; then
	POLYBENCH_C_DIR="$BUILD_DIR/PolyBenchC-4.2.1"
	mkdir -p "$POLYBENCH_C_DIR" || exit 1
	if [ -d "$POLYBENCH_C_DIR/.git" ]; then
		( cd "$POLYBENCH_C_DIR" && git pull )
	else
		git clone "https://github.com/jiriklepl/PolyBenchC-4.2.1.git" "$POLYBENCH_C_DIR"
	fi
fi

run_script() {
    if [ "$USE_SLURM" -eq 1 ]; then
        srun -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" --exclusive -w"$SLURM_WORKER" -t "$SLURM_TIMEOUT" --mem=0 "$@"
    else
        "$@"
    fi
}

( cd "$POLYBENCH_C_DIR" && run_script ./build.sh ) || exit 1
( cd . && run_script ./build.sh ) || exit 1

mkdir -p "$DATA_DIR"

find "$BUILD_DIR" -maxdepth 1 -executable -type f |
while read -r file; do
    filename=$(basename "$file")

    echo "collecting $filename"
    ( run_script ./run_noarr_algorithm.sh "Noarr" "$BUILD_DIR/$filename" & wait ) > "$DATA_DIR/$filename.log"
	echo "" >> "$DATA_DIR/$filename.log"
    ( run_script ./run_c_algorithm.sh "Baseline" "$POLYBENCH_C_DIR/$BUILD_DIR/$filename" & wait ) >> "$DATA_DIR/$filename.log"
    echo "done"
done
