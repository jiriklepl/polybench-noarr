#!/bin/bash -ex

NUM_RUNS=100

mkdir -p build

SET_CONFIG="-DEXTRALARGE_DATASET -DDATA_TYPE_IS_FLOAT"

(
	cd build

	# Build autotuners
	cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_CXX_FLAGS="$SET_CONFIG "
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
		config=$(grep -oE '"\w+"\s*:\s*[^,}]+' ../build/mmm_final_config.json | sed -E 's/"|://g' | awk '{printf("%s", "-DNOARR_PARAMETER_VALUE_" $1 "=" $2 " ")}END{print ""}')

		echo "$config" > "$filename-config"

		# build the autotuned version of the program
		cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_CXX_FLAGS="$SET_CONFIG -DNOARR_PASS_BY_DEFINE $config"
		cmake --build . -j"$(nproc)" -t "$(echo "$filename" | sed -E 's/_autotune//')"
	)
done
