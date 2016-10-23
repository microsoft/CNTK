#!/bin/bash
#
# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# Log steps, stop on error
set -x -e -o pipefail

USAGE="Usage: $0 <url-of-cntk-python-wheel> [--force]"
CNTK_PIP_URL=${1?$USAGE}
FORCE=$(! [ "$2" = "--force" ]; echo $?)

# Change to the script's directory
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"

# Check for tested OS (note: only a warning, we can live with lsb-release not being available)
[[ "$(lsb_release -i)" =~ :.*Ubuntu ]] && [[ "$(lsb_release -r)" =~ :.*(14\.04|16\.04) ]] || {
  printf "WARNING: this script was only tested on Ubuntu 14.04 and 16.04, installation may fail.\n"
}

###################
# Package installs

# Anaconda download / install dependencies
# [coreutils for sha{1,256}sum]
PACKAGES="bzip2 wget coreutils"

# CNTK examples dependencies
PACKAGES+=" git ca-certificates"

# CNTK run-time dependencies (OpenMPI)
if [[ "$(lsb_release -i)" =~ :.*Ubuntu ]] && [[ "$(lsb_release -r)" =~ :.*14\.04 ]]; then
  # On Ubuntu 14.04: need to build ourselves, openmpi-bin is too old
  BUILD_OPENMPI=1
  PACKAGES+=" wget ca-certificates build-essential"
else
  # Else: try with openmpi-bin
  BUILD_OPENMPI=0
  PACKAGES+=" openmpi-bin"
fi

if dpkg -s $PACKAGES 1>/dev/null 2>/dev/null; then
  printf "Packages already installed, skipping.\n"
else
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends $PACKAGES
fi

# Check URL early on for feedback
wget --spider -q "$CNTK_PIP_URL" || {
  printf "Please double-check URL '%s', does it exist?\n$USAGE" "$CNTK_PIP_URL"
  exit 1
}

#########################################
# On Ubuntu 14.04: OpenMPI build

if [ "$BUILD_OPENMPI" = "1" ]; then
  OPENMPI_PREFIX="$HOME/openmpi"
  if [ -d "$OPENMPI_PREFIX" ]; then
    printf "Path '%s' already exists, skipping OpenMPI build\n" "$OPENMPI_PREFIX"
  else
    OPENMPI_MAJOR_MINOR_VERSION=1.10
    OPENMPI_PATCH_VERSION=4
    OPENMPI_SHA1=1676a7da6cc8cde1d46f6296f38d575249b46cd9
    OPENMPI_VERSION=$OPENMPI_MAJOR_MINOR_VERSION.$OPENMPI_PATCH_VERSION
    OPENMPI=openmpi-$OPENMPI_VERSION
    wget --continue --no-verbose https://www.open-mpi.org/software/ompi/v$OPENMPI_MAJOR_MINOR_VERSION/downloads/$OPENMPI.tar.bz2
    echo "$OPENMPI_SHA1  $OPENMPI.tar.bz2" | sha1sum -c --strict -
    tar -xjf $OPENMPI.tar.bz2
    cd $OPENMPI
    ./configure --prefix=$OPENMPI_PREFIX
    make -j $(nproc) install
    cd ..
    rm -rf $OPENMPI
  fi
fi

#########################################
# Anaconda install and environment setup
# TODO consider miniconda

ANACONDA_PREFIX="$HOME/anaconda3"
if [ -d "$ANACONDA_PREFIX" ]; then
  printf "Path '%s' already exists, skipping Anaconda install\n" "$ANACONDA_PREFIX"
else
  ANACONDA=Anaconda3-4.1.1-Linux-x86_64.sh
  ANACONDA_SHA256=4f5c95feb0e7efeadd3d348dcef117d7787c799f24b0429e45017008f3534e55
  # --continue: use existing file
  wget --continue --no-verbose --no-check-certificate "https://repo.continuum.io/archive/$ANACONDA"
  echo "$ANACONDA_SHA256  $ANACONDA" | sha256sum -c --strict -
  chmod a+x "$ANACONDA"
  "./$ANACONDA" -b -p "$ANACONDA_PREFIX"
fi

CONDA="$HOME/anaconda3/bin/conda"
[ -x "$CONDA" ]
ACTIVATE="$HOME/anaconda3/bin/activate"
[ -x "$ACTIVATE" ]

CNTK_PY34_ENV_PREFIX="$ANACONDA_PREFIX/envs/cntk-py34"
if [ -d "$CNTK_PY34_ENV_PREFIX" ]; then
  printf "Path '%s' already exists, skipping CNTK Python 3.4 environment setup\n" "$CNTK_PY34_ENV_PREFIX"
else
  CNTK_PY34_ENV_FILE=conda-linux-cntk-py34-environment.yml

  cat >| "$CNTK_PY34_ENV_FILE" <<CONDAENV
name: cntk-py34
dependencies:
- matplotlib=1.5.3=np111py34_0
- jupyter=1.0.0=py34_3
- numpy=1.11.1=py34_0
- python=3.4.4=5
- scipy=0.18.1=np111py34_0
- pip:
  - pytest==3.0.2
  - sphinx==1.4.6
  - sphinx-rtd-theme==0.1.9
  - twine==1.8.1
CONDAENV

  # (--force shouldn't be needed)
  "$CONDA" env create --quiet --force --file "$CNTK_PY34_ENV_FILE" --prefix "$CNTK_PY34_ENV_PREFIX"

fi

###########################################
# Install CNTK module

set +x
source "$ACTIVATE" "$CNTK_PY34_ENV_PREFIX"
set -x

CNTK_MODULE_DIR="$CNTK_PY34_ENV_PREFIX/lib/python3.4/site-packages/cntk"

if [ -e "$CNTK_MODULE_DIR" ]; then
  if [ $FORCE = 1 ]; then
    printf "Removing previously installed CNTK module\n"
    pip uninstall --yes cntk

    pip install "$CNTK_PIP_URL"
  else
    printf "There is already a CNTK module installed, and --force was not specified, skipping Pip installation.\n"
  fi
else
  pip install "$CNTK_PIP_URL"
fi

###########################################
# Clone CNTK repository

CNTK_WORKING_COPY="$HOME/repos/cntk"

if [ -d "$CNTK_WORKING_COPY" ]; then
  printf "Path '%s' already exists, skipping CNTK clone\n" "$CNTK_WORKING_COPY"
else
  mkdir -p "$HOME/repos"
  CNTK_GIT_CLONE_URL=https://github.com/Microsoft/CNTK.git
  git clone --recursive "$CNTK_GIT_CLONE_URL" "$HOME/repos/cntk"
fi

OPENMPI_MESSAGE=
if [ "$BUILD_OPENMPI" = "1" ]; then
  OPENMPI_MESSAGE=$(printf "\n  export LD_LIBRARY_PATH=%s:\$LD_LIBRARY_PATH" "$OPENMPI_PREFIX/lib")
fi

cat <<FINALMESSAGE

************************************************************
CNTK v2 Python install complete.

To activate the CNTK v2 Python environment, run
$OPENMPI_MESSAGE
  source "$ACTIVATE" "$CNTK_PY34_ENV_PREFIX"

Please checkout examples in the CNTK repository clone here:

  $CNTK_WORKING_COPY/bindings/python/examples

************************************************************
FINALMESSAGE

# vim:set expandtab shiftwidth=2 tabstop=2:
