#!/bin/bash -ex

NUM_RUNS=${NUM_RUNS:-25}

mkdir -p build

BUILD_DIR=${BUILD_DIR:-build}
DATASET_SIZE=${DATASET_SIZE:-EXTRALARGE}
DATA_TYPE=${DATA_TYPE:-FLOAT}
NOARR_STRUCTURES_BRANCH=${NOARR_STRUCTURES_BRANCH:-tuning}

./build.sh || exit 1

mkdir -p autotuned

for file in "$BUILD_DIR"/*_autotune; do
	filename=$(basename "$file")

	(
		cd "$BUILD_DIR"

		# find the best configuration using opentuner
		python <("./$filename") --test-limit="$NUM_RUNS" --no-dups
	)

	(
		cd autotuned

		# transform the configuration into a list of defines
		config=$(grep -oE '"\w+"\s*:\s*[^,}]+' ../$BUILD_DIR/mmm_final_config.json | sed -E 's/"|://g' | awk '{printf("%s", "-DNOARR_PARAMETER_VALUE_" $1 "=" $2 " ")}END{print ""}')

		mv "../$BUILD_DIR/mmm_final_config.json" "$filename.config.json"
		echo "$config" > "$filename.config.txt"

		# build the autotuned version of the program
		cmake -DCMAKE_BUILD_TYPE=Release \
			-DNOARR_STRUCTURES_BRANCH="$NOARR_STRUCTURES_BRANCH" \
			-DCMAKE_CXX_FLAGS="-D${DATASET_SIZE}_DATASET -DDATA_TYPE_IS_$DATA_TYPE -DNOARR_PASS_BY_DEFINE $config" \
			..

		cmake --build . -j"$(nproc)" -t "$(echo "$filename" | sed -E 's/_autotune//')"
	)
done
