#!/bin/bash
#
# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"

PARSED_ARGS=$(getopt -o '' --long py-version:,anaconda-basepath: -n "$SCRIPT_NAME" -- "$@")

[ $? != 0 ] && {
  echo Terminating...
  exit 1
}

eval set -- "$PARSED_ARGS"
PY_VERSION=35
ANACONDA_PREFIX="$HOME/anaconda3"

while true; do
  case "$1" in
    --py-version)
      case "$2" in
        27 | 34 | 35)
          PY_VERSION="$2"
          ;;
        *)
          echo Invalid value for --py-version option, please specify 27, 34, or 35.
          exit 1
          ;;
      esac
      shift 2
      ;;
    --anaconda-basepath)
      ANACONDA_PREFIX="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
  esac
done

[ $# = 0 ] || {
  echo Extra parameters detected: $*
  exit 1
}

# Log steps, stop on error
# TODO cut down on logging
set -x -e -o pipefail

# Go to the drop root
cd "$SCRIPT_DIR/../../.."

PYWHEEL_QUALIFIER=cp$PY_VERSION-cp${PY_VERSION}m
[ $PY_VERSION = 27 ] && PYWHEEL_QUALIFIER+=u

CNTK_BIN_PATH="$PWD/cntk/bin"
CNTK_LIB_PATH="$PWD/cntk/lib"
CNTK_DEP_LIB_PATH="$PWD/cntk/dependencies/lib"
CNTK_EXAMPLES_PATH="$PWD/Examples"
CNTK_TUTORIALS_PATH="$PWD/Tutorials"
CNTK_BINARY="$CNTK_BIN_PATH/cntk"
CNTK_PY_ENV_FILE="$SCRIPT_DIR/conda-linux-cntk-py$PY_VERSION-environment.yml"
CNTK_WHEEL_PATH="cntk/python/cntk-2.0.beta12.0-$PYWHEEL_QUALIFIER-linux_x86_64.whl"

test -d "$CNTK_BIN_PATH" && test -d "$CNTK_LIB_PATH" && test -d "$CNTK_DEP_LIB_PATH" &&
test -d "$CNTK_TUTORIALS_PATH" &&
test -d "$CNTK_EXAMPLES_PATH" && test -x "$CNTK_BINARY" &&
test -f "$CNTK_PY_ENV_FILE" && test -f "$CNTK_WHEEL_PATH" || {
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

CONDA="$ANACONDA_PREFIX/bin/conda"
[ -x "$CONDA" ]
PY_ACTIVATE="$ANACONDA_PREFIX/bin/activate"
[ -x "$PY_ACTIVATE" ]
PY_DEACTIVATE="$ANACONDA_PREFIX/bin/deactivate"
[ -x "$PY_DEACTIVATE" ]

CNTK_PY_ENV_NAME="cntk-py$PY_VERSION"
CNTK_PY_ENV_PREFIX="$ANACONDA_PREFIX/envs/$CNTK_PY_ENV_NAME"
if [ -d "$CNTK_PY_ENV_PREFIX" ]; then
  "$CONDA" env update --file "$CNTK_PY_ENV_FILE" --name "$CNTK_PY_ENV_NAME" || {
    echo Updating Anaconda environment failed.
    exit 1
  }
else
  "$CONDA" env create --file "$CNTK_PY_ENV_FILE" --prefix "$CNTK_PY_ENV_PREFIX" || {
    echo Creating Anaconda environment failed.
    rm -rf "$CNTK_PY_ENV_PREFIX"
    exit 1
  }
fi

###########################################
# Install CNTK module

set +x
source "$PY_ACTIVATE" "$CNTK_PY_ENV_PREFIX"
set -x

pip install "$CNTK_WHEEL_PATH"

set +x
source "$PY_DEACTIVATE"
set -x

###########################################
# Create an activation script

LD_LIBRARY_PATH_SETTING="$CNTK_LIB_PATH:$CNTK_DEP_LIB_PATH"
if [ "$BUILD_OPENMPI" = "1" ]; then
  LD_LIBRARY_PATH_SETTING+=":$OPENMPI_PREFIX/lib"
fi
LD_LIBRARY_PATH_SETTING+=":\$LD_LIBRARY_PATH"

ACTIVATE_SCRIPT_NAME=activate-cntk
cat >| "$ACTIVATE_SCRIPT_NAME" <<ACTIVATE
if [ -z "\$BASH_VERSION" ]; then
  echo Error: only Bash is supported.
elif [ "\$(basename "\$0" 2> /dev/null)" == "$ACTIVATE_SCRIPT_NAME" ]; then
  echo Error: this script is meant to be sourced. Run 'source activate-cntk'
else
  export PATH="$CNTK_BIN_PATH:\$PATH"
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_SETTING"
  source "$PY_ACTIVATE" "$CNTK_PY_ENV_PREFIX"

  cat <<MESSAGE

************************************************************
CNTK is activated.

Please checkout tutorials and examples here:
  $CNTK_TUTORIALS_PATH
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

Please checkout tutorials and examples here:
  $CNTK_TUTORIALS_PATH
  $CNTK_EXAMPLES_PATH

************************************************************
FINALMESSAGE

# vim:set expandtab shiftwidth=2 tabstop=2:
