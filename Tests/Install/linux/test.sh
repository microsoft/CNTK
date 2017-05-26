#!/bin/bash
set -x -e -o pipefail

USAGE="Usage: $0 [--py-version [27|34|35|36]] -- <drops-to-test>"

SCRIPT_DIR="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"

PY_VERSION=35
WHEEL_BASE_URL=https://cntk.ai/PythonWheel/

while [ $# -gt 0 ]; do
  case "$1" in
    --py-version)
      case "$2" in
        27 | 34 | 35 | 36)
          PY_VERSION="$2"
          ;;
        *)
          echo Invalid or missing value for --py-version option, please specify 27, 34, 35, or 36.
          exit 1
          ;;
      esac
      shift # extra shift
      ;;
    --wheel-base-url)
      WHEEL_BASE_URL="$2"
      shift # extra shift
      ;;
    --)
      # Stop processing options
      shift
      break
      ;;
    *)
      echo Unknown option $1
      exit 1
      ;;
  esac
  shift
done

if [ -z "$1" ]; then
  echo $USAGE
  exit 1
fi

DROP_RESERVED=BinaryDrop.tar.gz

# (There's extra overhead by having several images in Docker build context, but well)

for drop in $*; do
  [ -f "$drop" ]
  [[ "$drop" == *tar.gz ]]
  DROP_DIR="$(dirname "$(readlink -f "$drop")")"
  [ "$?" = 0 ]
  DROP_FILE="$(basename "$(readlink -f "$drop")")"
  [ "$?" = 0 ]
  [ "$DROP_DIR" = "$SCRIPT_DIR" ]
  [ "$DROP_FILE" != "$DROP_RESERVED" ]
done

for drop in $*; do

  DROP_FILE="$(basename "$(readlink -f "$drop")")"

  if [[ "$DROP_FILE" == *CPU* ]] || [[ "$DROP_FILE" == *cpu* ]]; then
    TEST_DEVICE=cpu
    DOCKER_TO_RUN=docker
    DOCKERFILE_SUFFIX=CPU
  else
    TEST_DEVICE=gpu
    DOCKER_TO_RUN=nvidia-docker
    DOCKERFILE_SUFFIX=GPU
  fi

  rm -f "$DROP_RESERVED"

  ln -s "$DROP_FILE" "$DROP_RESERVED"

  IMAGE=cntk:installtest
  for base in Ubuntu16 Ubuntu14; do
    docker build --build-arg PY_VERSION=$PY_VERSION --build-arg WHEEL_BASE_URL=$WHEEL_BASE_URL -t $IMAGE -f Dockerfile-$base-$DOCKERFILE_SUFFIX .
    $DOCKER_TO_RUN run --rm $IMAGE su - testuser -c "./run-test.sh $TEST_DEVICE"
    docker rmi $IMAGE
  done

done

echo Note: you may want to clean up Docker images.

# vim:set expandtab shiftwidth=2 tabstop=2:
