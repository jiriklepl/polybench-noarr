#!/usr/bin/env bash

# This script compares the output of the C and C++ & Noarr implementations of the Polybench benchmarks
# It assumes that the C++ & Noarr implementations are built in the build directory and that the C implementations are built in the $POLYBENCH_C_DIR/build directory

if [ -z "$POLYBENCH_C_DIR" ]; then
	echo "POLYBENCH_C_DIR is not set" >&2
	exit 1
fi

dirname=$(mktemp -d)

trap "rm -rf $dirname" EXIT

(
	cd "$POLYBENCH_C_DIR" || exit 1
	./build.sh || exit 1
)

(
	./build.sh || exit 1
)

find build -maxdepth 1 -executable -type f \
	| while read -r file; do
		filename=$(basename "$file")

		case "$filename" in
			*_autotune)
				continue
				;;
		esac

		grep -vE '^\s*#|^\s*$' compare-ignore.txt | grep "^$filename\$" > /dev/null && continue

		echo "Comparing $filename"

		printf "Noarr:             " >&2
		"build/$filename" > "$dirname/cpp"

		if [ -f "autotuned/$filename" ]; then
			printf "Noarr (autotuned): " >&2
			"autotuned/$filename" > "$dirname/cpp-autotuned"
		fi

		printf "C:                 " >&2
		"$POLYBENCH_C_DIR/build/$filename" 1>&2 2> "$dirname/c"

		diff -y --suppress-common-lines \
			<(grep -oE '[0-9]+(\.[0-9]+)?' "$dirname/cpp" | cat -n) \
			<(grep -oE '[0-9]+\.[0-9]+' "$dirname/c" | cat -n) || \
				printf "Different output on %s\n" "$filename" >&2

		if [ -f "autotuned/$filename" ]; then
			diff -y --suppress-common-lines \
				<(grep -oE '[0-9]+(\.[0-9]+)?' "$dirname/cpp-autotuned" | cat -n) \
				<(grep -oE '[0-9]+\.[0-9]+' "$dirname/c" | cat -n) || \
					printf "Different output on %s (autotuned)\n" "$filename" >&2
		fi
	done
