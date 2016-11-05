#!/bin/bash
#
# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# Log steps, stop on error
# TODO cut down on logging
set -x -e -o pipefail

REPO_TAG=v2.0.beta2.0

while [ $# -gt 0 ]; do
  case "$1" in
    --repo-tag)
      REPO_TAG="$2"
      [ -z "$REPO_TAG" ] && {
        echo Missing value for --repo-tag option.
        exit 1
      }
      shift # extra shift
      ;;
    *)
      echo Unknown option $1
      exit 1
      ;;
  esac
  shift
done

SCRIPT_DIR="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"

# Go to the drop root
cd "$SCRIPT_DIR/../.."

CNTK_BIN_PATH="$PWD/cntk/bin"
CNTK_LIB_PATH="$PWD/cntk/lib"
CNTK_DEP_LIB_PATH="$PWD/cntk/dependencies/lib"
CNTK_EXAMPLES_PATH="$PWD/Examples"
CNTK_BINARY="$CNTK_BIN_PATH/cntk"
CNTK_PY34_ENV_FILE="$SCRIPT_DIR/conda-linux-cntk-py34-environment.yml"
CNTK_WHEEL_PATH="cntk/python/cntk-2.0.beta2.0-cp34-cp34m-linux_x86_64.whl"
test -d "$CNTK_BIN_PATH" && test -d "$CNTK_LIB_PATH" && test -d "$CNTK_DEP_LIB_PATH" && 
test -d "$CNTK_EXAMPLES_PATH" && test -x "$CNTK_BINARY" &&
test -f "$CNTK_PY34_ENV_FILE" && test -f "$CNTK_WHEEL_PATH" || {
  echo Cannot find expected drop content. Please double-check that this is a
  echo CNTK binary drop for Linux. Go to https://github.com/Microsoft/CNTK/wiki
  echo for help.
  exit 1
}

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

# Additional packages for ImageReader
PACKAGES+=" libjasper1 libjpeg8 libpng12-0"

if dpkg -s $PACKAGES 1>/dev/null 2>/dev/null; then
  printf "Packages already installed, skipping.\n"
else
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends $PACKAGES
fi

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
PY_ACTIVATE="$HOME/anaconda3/bin/activate"
[ -x "$PY_ACTIVATE" ]

CNTK_PY34_ENV_PREFIX="$ANACONDA_PREFIX/envs/cntk-py34"
if [ -d "$CNTK_PY34_ENV_PREFIX" ]; then
  printf "Path '%s' already exists, skipping CNTK Python 3.4 environment setup\n" "$CNTK_PY34_ENV_PREFIX"
else
  # (--force shouldn't be needed)
  "$CONDA" env create --quiet --force --file "$CNTK_PY34_ENV_FILE" --prefix "$CNTK_PY34_ENV_PREFIX" || {
    echo Creating Anaconda environment failed.
    rm -rf "$CNTK_PY34_ENV_PREFIX"
    exit 1
  }
fi

###########################################
# Install CNTK module

set +x
source "$PY_ACTIVATE" "$CNTK_PY34_ENV_PREFIX"
set -x

pip install "$CNTK_WHEEL_PATH"

###########################################
# Clone CNTK repository

CNTK_WORKING_COPY="$HOME/repos/cntk"

if [ -d "$CNTK_WORKING_COPY" ]; then
  printf "Path '%s' already exists, skipping CNTK clone\nMake sure to checkout $REPO_TAG\n" "$CNTK_WORKING_COPY"
else
  mkdir -p "$HOME/repos"
  CNTK_GIT_CLONE_URL=https://github.com/Microsoft/CNTK.git
  git clone --branch $REPO_TAG --recursive "$CNTK_GIT_CLONE_URL" "$HOME/repos/cntk"
fi

LD_LIBRARY_PATH_SETTING="$CNTK_LIB_PATH:$CNTK_DEP_LIB_PATH"
if [ "$BUILD_OPENMPI" = "1" ]; then
  LD_LIBRARY_PATH_SETTING+=":$OPENMPI_PREFIX/lib"
fi
LD_LIBRARY_PATH_SETTING+=":\$LD_LIBRARY_PATH"

###########################################
# Create an activation script

ACTIVATE_SCRIPT_NAME=activate-cntk
cat >| "$ACTIVATE_SCRIPT_NAME" <<ACTIVATE
if [ -z "\$BASH_VERSION" ]; then
  echo Error: only Bash is supported.
elif [ "\$(basename "\$0" 2> /dev/null)" == "$ACTIVATE_SCRIPT_NAME" ]; then
  echo Error: this script is meant to be sourced. Run 'source activate-cntk'
else
  export PATH="$CNTK_BIN_PATH:\$PATH"
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_SETTING"
  source "$PY_ACTIVATE" "$CNTK_PY34_ENV_PREFIX"

  cat <<MESSAGE

************************************************************
CNTK is activated.

Please check out CNTK Python examples in the CNTK repository clone here:

  $CNTK_WORKING_COPY/bindings/python/examples

Please check out CNTK Brainscript examples here:

  $CNTK_EXAMPLES_PATH

************************************************************
MESSAGE

fi


ACTIVATE

cat <<FINALMESSAGE

************************************************************
CNTK install complete.

To activate the CNTK environment, run
  source "$PWD/$ACTIVATE_SCRIPT_NAME"

Please check out CNTK Python examples in the CNTK repository clone here:

  $CNTK_WORKING_COPY/bindings/python/examples

Please check out CNTK Brainscript examples here:

  $CNTK_EXAMPLES_PATH

************************************************************
FINALMESSAGE

# vim:set expandtab shiftwidth=2 tabstop=2:
