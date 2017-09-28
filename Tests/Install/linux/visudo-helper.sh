#!/bin/bash
set -x
FILE="$1"
[ "$FILE" = "--" ] && FILE="$2"
printf "testuser\tALL=(ALL) NOPASSWD: ALL\n" >> "$FILE"
# use as: VISUAL=/path/to/script visudo
