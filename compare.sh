#!/bin/bash

set -eo pipefail

# This script compares the output of the C and C++/Noarr implementations of the Polybench benchmarks
# It assumes that the C++/Noarr implementations are built in the build directory and that the C implementations are built in the $POLYBENCH_C_DIR/build directory

export BUILD_DIR=${BUILD_DIR:-build}
export SKIP_DIFF=${SKIP_DIFF:-0}
export ALGORITHM=${ALGORITHM:-}

if [ -z "$POLYBENCH_C_DIR" ]; then
	echo "POLYBENCH_C_DIR is not set" >&2
	exit 1
fi

dirname=$(mktemp -d)

cleanup() {
	echo "deleting $dirname" >&2
	rm -rf "$dirname"
}

trap cleanup EXIT

( cd "$POLYBENCH_C_DIR" && ./build.sh )
( cd . && ./build.sh )

find "$BUILD_DIR" -maxdepth 1 -executable -type f |
while read -r file; do
	filename=$(basename "$file")

	if [ -n "$ALGORITHM" ]; then
		case "$filename" in
			"$ALGORITHM")
				;;
			*)
				continue
				;;
		esac
	fi

	echo "Comparing $filename"

	printf "\tNoarr:             "
	"$BUILD_DIR/$filename" 2>&1 1> "$dirname/cpp"

	printf "\tBaseline:          "
	"$POLYBENCH_C_DIR/$BUILD_DIR/$filename" 2> "$dirname/c"

	if [ "$SKIP_DIFF" -eq 1 ]; then
		continue
	fi

	grep -woE '[0-9\.]+|nan' "$dirname/c" > "$dirname/c.tmp" && mv "$dirname/c.tmp" "$dirname/c"
	grep -woE '[0-9\.]+|nan' "$dirname/cpp" > "$dirname/cpp.tmp" && mv "$dirname/cpp.tmp" "$dirname/cpp"

	# skip files that are exactly the same
	if cmp -s "$dirname/c" "$dirname/cpp"; then
		continue
	fi

	paste "$dirname/c" "$dirname/cpp" |
	awk "BEGIN {
		different = 0
		n = 0
		changes = 0
	}

	{
		n++
		if (\$1 != \$2 && changes < 10) {
			print \"baseline\", n, \$1
			print \"   noarr\", n, \$2
			changes++
			different = 1
		}

		if (changes >= 10)
			nextfile
		next
	}

	{ different = 1; nextfile }

	END {
		if (different) {
			printf \"Different output on %s \n\", \"$filename\"
			exit 1
		}
	}" 1>&2
done
