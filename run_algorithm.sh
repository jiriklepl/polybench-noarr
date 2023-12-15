#!/bin/shell

if [ -z "$POLYBENCH_C_DIR" ]; then
	POLYBENCH_C_DIR="$BUILD_DIR/PolyBenchC-4.2.1"
fi

FILE="$1"

filename=$(basename "$FILE")


printf "\tNoarr:             "
"$BUILD_DIR/$filename" 2>&1 1>/dev/null

for _ in $(seq 10); do
    printf "\tNoarr:             "
    "$BUILD_DIR/$filename" 2>&1 1>/dev/null
done

printf "\tC:                 "
"$POLYBENCH_C_DIR/$BUILD_DIR/$filename" 2>/dev/null

for _ in $(seq 10); do
    printf "\tNoarr:             "
    "$POLYBENCH_C_DIR/$BUILD_DIR/$filename" 2>/dev/null
done

