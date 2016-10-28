#!/bin/bash
set -e -x

# TODO extend matrix

IMAGE=cntk:installtest
for base in Ubuntu16 Ubuntu14; do
  docker build -t $IMAGE -f Dockerfile-$base-GPU .
  nvidia-docker run --rm $IMAGE su - testuser -c "./run-test.sh gpu"
  #docker rmi $IMAGE
done
