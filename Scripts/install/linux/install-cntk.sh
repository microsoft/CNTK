#!/bin/bash
#
# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"

PARSED_ARGS=$(getopt -o '' --long py-version:,anaconda-basepath:,wheel-base-url:,docker -n "$SCRIPT_NAME" -- "$@")

function die {
  set +x
  echo -e $1
  echo Go to https://github.com/Microsoft/CNTK/wiki/Setup-Linux-Binary-Script for help.
  exit 1
}

[ $? != 0 ] && die "Terminating..."

eval set -- "$PARSED_ARGS"
PY_VERSION=35
ANACONDA_PREFIX="$HOME/anaconda3"
WHEEL_BASE_URL=https://cntk.ai/PythonWheel/
DOCKER_INSTALLATION=0

while true; do
  case "$1" in
    --py-version)
      case "$2" in
        27 | 34 | 35)
          PY_VERSION="$2"
          ;;
        *)
          die "Invalid value for --py-version option, please specify 27, 34, or 35."
          ;;
      esac
      shift 2
      ;;
    --anaconda-basepath)
      ANACONDA_PREFIX="$2"
      shift 2
      ;;
    --docker) # use during Docker Hub image building, not documented
      DOCKER_INSTALLATION=1
      shift
      ;;
    --wheel-base-url) # intended for testing, not documented
      WHEEL_BASE_URL="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
  esac
done

[ $# = 0 ] || die "Extra parameters detected: $*"

# Log steps, stop on error
# TODO cut down on logging
set -x -e -o pipefail

# Go to the drop root
cd "$SCRIPT_DIR/../../.."

CNTK_VERSION_PATH="version.txt"
CNTK_BIN_PATH="$PWD/cntk/bin"
CNTK_LIB_PATH="$PWD/cntk/lib"
CNTK_DEP_LIB_PATH="$PWD/cntk/dependencies/lib"
CNTK_EXAMPLES_PATH="$PWD/Examples"
CNTK_TUTORIALS_PATH="$PWD/Tutorials"
CNTK_BINARY="$CNTK_BIN_PATH/cntk"
CNTK_PY_ENV_FILE="$SCRIPT_DIR/conda-linux-cntk-py$PY_VERSION-environment.yml"

test -f "$CNTK_VERSION_PATH" &&
test -d "$CNTK_BIN_PATH" && test -d "$CNTK_LIB_PATH" && test -d "$CNTK_DEP_LIB_PATH" &&
test -d "$CNTK_TUTORIALS_PATH" &&
test -d "$CNTK_EXAMPLES_PATH" && test -x "$CNTK_BINARY" &&
test -f "$CNTK_PY_ENV_FILE" ||
  die "Cannot find expected drop content. Please double-check that this is a CNTK binary drop for Linux."

# Check for tested OS (note: only a warning, we can live with lsb-release not being available)
[[ "$(lsb_release -i)" =~ :.*Ubuntu ]] && [[ "$(lsb_release -r)" =~ :.*(14\.04|16\.04) ]] || {
  printf "WARNING: this script was only tested on Ubuntu 14.04 and 16.04, installation may fail.\n"
}

readarray -t versionInfo < "$CNTK_VERSION_PATH" ||
  die "Unable to read version file '$CNTK_VERSION_PATH'."

[[ ${versionInfo[0]} =~ ^CNTK-([1-9][0-9a-z-]*)$ ]] ||
  die "Malformed version information in version file, ${versionInfo[0]}."

DASHED_VERSION="${BASH_REMATCH[1]}"
DOTTED_VERSION="${DASHED_VERSION//-/.}"

[[ ${versionInfo[2]} =~ ^(GPU|CPU-Only|GPU-1bit-SGD)$ ]] ||
  die "Malformed target configuration file, ${versionInfo[2]}."

TARGET_CONFIGURATION="${BASH_REMATCH[1]}"

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
# Check Python Wheel availability

PYWHEEL_QUALIFIER=cp$PY_VERSION-cp${PY_VERSION}m
[ $PY_VERSION = 27 ] && PYWHEEL_QUALIFIER+=u
CNTK_WHEEL_NAME="cntk-$DOTTED_VERSION-$PYWHEEL_QUALIFIER-linux_x86_64.whl"
CNTK_WHEEL_PATH="cntk/python/$CNTK_WHEEL_NAME"

# Check online if there is no wheel locally
if ! test -f "$CNTK_WHEEL_PATH"; then
  CNTK_WHEEL_PATH="$WHEEL_BASE_URL/$TARGET_CONFIGURATION/$CNTK_WHEEL_NAME"

  wget -q --spider "$CNTK_WHEEL_PATH" ||
    die "Python wheel not available locally and cannot reach $CNTK_WHEEL_PATH for Python\nwheel installation online. Please double-check Internet connectivity."

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
    if [ "$DOCKER_INSTALLATION" = "1" ]; then
      rm -f $OPENMPI.tar.bz2
    fi
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
  if [ "$DOCKER_INSTALLATION" = "1" ]; then
    rm -f $ANACONDA
  fi
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
  "$CONDA" env update --file "$CNTK_PY_ENV_FILE" --name "$CNTK_PY_ENV_NAME" ||
    die "Updating Anaconda environment failed."
else
  "$CONDA" env create --file "$CNTK_PY_ENV_FILE" --prefix "$CNTK_PY_ENV_PREFIX" || {
    rm -rf "$CNTK_PY_ENV_PREFIX"
    die "Creating Anaconda environment failed."
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

To deactivate the environment run

  source $PY_DEACTIVATE

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

To deactivate the environment run

  source $PY_DEACTIVATE

************************************************************
FINALMESSAGE

if [ "$DOCKER_INSTALLATION" = "1" ]; then
  # Docker Hub Image specific actions

  # Clean up
  apt-get -y autoremove
  rm -rf /var/lib/apt/lists/*
  "$ANACONDA_PREFIX/bin/conda" clean --all --yes
  # Remove Python Wheels "just in case"
  # As of v2.0 Beta 15 they should not be a part of the Drop
  rm -rf ./cntk/python

  # Add login Welcome message
  # and call CNTK activation on login

  cat >> /root/.bashrc <<WELCOMEACTIVATECNTK
  # CNTK Welcome Message
  cat <<MESSAGE

************************************************************
Welcome to Microsoft Cognitive Toolkit (CNTK) v. $CNTK_VERSION

Activating CNTK environment...

(Use command below to activate manually when needed)

  source "$PWD/$ACTIVATE_SCRIPT_NAME"
MESSAGE

source "$PWD/$ACTIVATE_SCRIPT_NAME"
WELCOMEACTIVATECNTK

fi

# vim:set expandtab shiftwidth=2 tabstop=2:
