# Tag: nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
# Created: 2018-10-22T21:14:30.605789926Z
# Label: com.nvidia.cuda.version: 10.0.
# Label: com.nvidia.cudnn.version: 7.3.1.20
# Label: com.nvidia.nccl.version: 2.3.5
#
# To build, run from the parent with the command line:
#      docker build -t <image name> -f CNTK-GPU-Image/Dockerfile .

# Ubuntu 16.04.5
FROM nvidia/cuda@sha256:362e4e25aa46a18dfa834360140e91b61cdb0a3a2796c8e09dadb268b9de3f6b

ARG PY_VERSION
ARG WHEEL_BASE_URL

# Set up fake user / sudo environment:
RUN apt-get update && apt-get install -y --no-install-recommends sudo lsb-release make build-essential
RUN adduser --gecos "Test User" --disabled-password testuser && test -d /home/testuser
COPY visudo-helper.sh prep-run-test.sh /root/
RUN VISUAL=/root/visudo-helper.sh visudo

COPY test_wrapper.sh /home/testuser
COPY BinaryDrop.tar.gz /home/testuser
COPY GPU/ /home/testuser/GPU/

RUN chown -R testuser:testuser /home/testuser

# TODO run repeated
RUN su - testuser -c "./test_wrapper.sh BinaryDrop.tar.gz $PY_VERSION $WHEEL_BASE_URL"
RUN /root/prep-run-test.sh
