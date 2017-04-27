#!/bin/bash
#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
#
# Internal use: populate data sets to their default location

set -x -e -o pipefail

# Configuration data
declare -A dataSetMap
dataSetMap=(
  ["MNIST-v0.tar.gz"]="Examples/Image/DataSets/MNIST"
)

# Change directory to repository root
SCRIPT_DIR="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"
cd "$SCRIPT_DIR/.."

# Validate source directory
[ -n "$CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY" ]
if [ "$OS" == "Windows_NT" ]; then
  DATADIR="$(cygpath -au "$CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY")"
else
  DATADIR="$CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY"
fi

[ -d "$DATADIR" ]
DATADIR+=/DataSets
[ -d "$DATADIR" ]

# Un-tar datasets (note: tar should automatically uncompress)
for dataSet in "${!dataSetMap[@]}"; do
  archive="$DATADIR/$dataSet"
  outDir="${dataSetMap[$dataSet]}"
  [ -f "$archive" ]
  [ -d "$outDir" ]
  tar -xf "$archive" -C "$outDir"
done
