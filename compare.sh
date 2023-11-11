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

		# Uncomment the following line to compare only the subset of benchmarks that are autotuned
		# [ ! -f "autotuned/$filename" ] && continue

		echo "Comparing $filename"

		printf "Noarr:             " >&2
		"build/$filename" > "$dirname/cpp"

		if [ -f "autotuned/$filename" ]; then
			printf "Noarr (autotuned): " >&2
			"autotuned/$filename" > "$dirname/cpp-autotuned"
		fi

		printf "C:                 " >&2
		"$POLYBENCH_C_DIR/build/$filename" 1>&2 2> "$dirname/c"

		if [ -f "autotuned/$filename" ]; then
			paste <(grep -oE '[0-9]+\.[0-9]+' "$dirname/c") <(grep -oE '[0-9]+(\.[0-9]+)?' "$dirname/cpp") <(grep -oE '[0-9]+(\.[0-9]+)?' "$dirname/cpp-autotuned")
		else 
			paste <(grep -oE '[0-9]+\.[0-9]+' "$dirname/c") <(grep -oE '[0-9]+(\.[0-9]+)?' "$dirname/cpp")
		fi | awk "BEGIN {
			different = 0
			n = 0
			changes = 0
			autotune_changes = 0
			outputs = \"$([ -f "autotuned/$filename" ] && echo 3 || echo 2)\"
		}

		NF == outputs {
			n++
			if (\$1 != \$2 && changes < 10) {
				print \"baseline\", n, \$1
				print \"   noarr\", n, \$2
				changes++
				different = 1
			}
			if (outputs == 3 && \$1 != \$3 && autotune_changes < 10) {
				print \"baseline\", n, \$1
				print \"autotune\", n, \$3
				autotune_changes++
				different = 1
			}

			if (changes >= 10 && (outputs == 2 || autotune_changes >= 10))
				nextfile
			next
		}

		{ different = 1; nextfile }

		END {
			if (different) {
				printf \"Different output on %s \n\", \"$filename\"
				exit 1
			}
		}" >&2
	done
