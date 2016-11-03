#!/bin/bash
#
# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
#
# This script is used to run tests from a source checkout against local CNTK
# module and examples.
# Assumes you are already in an activated CNTK python environment.
#
# Note: this is close but not exactly the same as our CI tests. If you need
# exactly the same setup, install the CNTK python module from a wheel into a
# clean CNTK python environment, don't modify PYTHONPATH, are run the
# associated end-to-end tests via TestDriver.py.
#
# Note: currently hard-coded for GPU build and testing on GPU device.

set -e -x

SCRIPT_DIR="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/../../build/gpu/release/lib
export PYTHONPATH=$PWD:$PYTHONPATH/cntk

for d in cntk doc examples; do
  pushd $d
  pytest --deviceid gpu
  popd
done
