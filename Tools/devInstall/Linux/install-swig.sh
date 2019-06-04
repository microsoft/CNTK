#!/bin/bash
#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
#
# Build and install SWIG on Ubuntu 14.04.
# SWIG is a source build dependency for the CNTK Java and Python bindings.
#
# This script is meant to be run interactively, and requires root permission
# (e.g., via sudo) for installation of build dependencies as well as the final
# binary.

SWIG_VERSION=3.0.10
SWIG_PATH=swig-$SWIG_VERSION
SWIG_PREFIX=/usr/local/$SWIG_PATH

[[ -d $SWIG_PREFIX ]] && {
  echo There already is a directory $SWIG_PREFIX.
  echo In case you want to rebuild and install, remove it first.
  exit 1
}

set -e -x -o pipefail

if [[ $EUID -eq 0 ]]; then
  AS_ROOT=
else
  AS_ROOT=sudo
fi

REQUIRED_PACKAGES="wget ca-certificates build-essential libpcre3-dev"
if dpkg -s $REQUIRED_PACKAGES 1>/dev/null 2>/dev/null; then
  printf "Packages already installed, skipping.\n"
else
  $AS_ROOT apt-get update
  $AS_ROOT apt-get install -y --no-install-recommends $REQUIRED_PACKAGES
fi

SCRIPT_DIR="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"
cd "$SCRIPT_DIR"
wget -q https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/swig/$SWIG_VERSION/$SWIG_PATH.tar.gz -O - | tar xvfz -
cd swig-$SWIG_VERSION
# Note: we specify --without-alllang to suppress building tests and examples for specific languages.
./configure --prefix=$SWIG_PREFIX --without-perl5 --without-alllang
make -j $(nproc)
$AS_ROOT make install
cd ..
rm -rf "$SWIG_PATH"
