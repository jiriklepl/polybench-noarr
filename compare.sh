#!/usr/bin/env bash

# This script compares the output of the C and C++ & Noarr implementations of the Polybench benchmarks
# It assumes that the C++ & Noarr implementations are built in the build directory and that the C implementations are built in the $POLYBENCH_C_DIR/build directory

if [ -z "$POLYBENCH_C_DIR" ]; then
	echo "POLYBENCH_C_DIR is not set" >&2
	exit 1
fi

find build -maxdepth 1 -executable -type f \
	| while read -r file; do
		filename=$(basename "$file")
		echo "Comparing $filename"
		diff -y --suppress-common-lines <("build/$filename" | grep -oE '[0-9]+(\.[0-9]+)?' | cat -n) <("$POLYBENCH_C_DIR/build/$filename" 2>&1 | grep -oE '[0-9]+(\.[0-9]+)?' | cat -n) ||  printf "Different output on %s\n" "$filename" >&2
	done
