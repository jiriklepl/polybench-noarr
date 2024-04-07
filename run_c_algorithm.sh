#!/bin/bash

prefix="$1"
file="$2"

printf "\t%s: " $prefix
"$file" 2>/dev/null

for _ in $(seq 10); do
    printf "\t%s: " $prefix
    "$file" 2>/dev/null
done
