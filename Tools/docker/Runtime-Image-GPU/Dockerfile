#FROM library/ubuntu:14.04
FROM nvidia/cuda:7.5-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    # General and Python
        ca-certificates \
        libjasper-runtime \
        python-dev \
        python-numpy \
        python-pip \
        python-yaml \
        wget \
    # For Open MPI
        gcc \
        g++ \
        make \
    # For Pillow
        libfreetype6 \
        libfreetype6-dev \
        libjpeg62-dev \
        libjpeg8 \
    # For scipy
        gfortran \
        libatlas-base-dev \
        libblas-dev \
        liblapack-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# For Pillow
RUN apt-get update && apt-get build-dep -y --no-install-recommends \
        python-imaging \
        && \
    rm -rf /var/lib/apt/lists/*

# Install Pillow and scipy
RUN pip install Pillow
RUN pip install scipy

# Install Open MPI
RUN OPENMPI_VERSION=1.10.3 && \
    wget -q -O - https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/mpi && \
    make -j"$(nproc)" install && \
    rm -rf /openmpi-${OPENMPI_VERSION}

ENV PATH /usr/local/mpi/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/mpi/lib:$LD_LIBRARY_PATH

# Get CNTK Binary Distribution
RUN  wget -q -O - https://cntk.ai/BinaryDrop/CNTK-1-7-2-Linux-64bit-GPU.tar.gz | tar -xzf -

ENV PATH /cntk/cntk/bin:$PATH
ENV LD_LIBRARY_PATH /cntk/cntk/lib:/cntk/cntk/dependencies/lib:$LD_LIBRARY_PATH

# Clean-up
RUN apt-get update && apt-get -y autoremove
