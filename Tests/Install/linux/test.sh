#!/bin/bash
set -e -x

docker build -t cntkpyv2:cpuonly -f Dockerfile-Ubuntu14-CPU .
docker build -t cntkpyv2:cpuonly-ubuntu16 -f Dockerfile-CPU .

docker build -t cntkpyv2:gpu -f Dockerfile-Ubuntu14-GPU .
docker build -t cntkpyv2:gpu-ubuntu16 -f Dockerfile-GPU .
