#!/bin/bash -ex

NUM_RUNS=100

mkdir -p build

(
	cd build

	# Build autotuners
	cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_CXX_FLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -D_POSIX_C_SOURCE=200809L"
	cmake --build . -j"$(nproc)"
)

mkdir -p autotuned

for file in build/*_autotune; do
	filename=$(basename "$file")

	(
		cd build

		# find the best configuration using opentuner
		python <("./$filename") --test-limit="$NUM_RUNS" --no-dups
	)

	(
		cd autotuned

		# transform the configuration into a list of defines
		config=$(grep -oE '"\w+"\s*:\s*[^,}]+' ../build/mmm_final_config.json | sed -E 's/"|://g' | awk '{printf("%s", " -DNOARR_PARAMETER_VALUE_" $1 "=" $2)}END{print ""}')

		# build the autotuned version of the program
		cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_CXX_FLAGS="-DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -D_POSIX_C_SOURCE=200809L -DNOARR_PASS_BY_DEFINE $config"
		cmake --build . -j"$(nproc)" -t "$(echo "$filename" | sed -E 's/_autotune//')"
	)
done
