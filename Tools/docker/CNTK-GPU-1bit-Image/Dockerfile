# Tag: nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# Created: 2017-11-21T06:34:14.675603521Z
# Label: com.nvidia.build.id: 41212533
# Label: com.nvidia.build.ref: e0edb5359ecb7bd3d86f0c9bfa18c2260b741ebb
# Label: com.nvidia.cuda.version: 9.0.176
# Label: com.nvidia.cudnn.version: 7.0.4.31
# Label: com.nvidia.nccl.version: 2.1.2
# Ubuntu 16.04.5
FROM nvidia/cuda@sha256:33add9c50ab76b8f3a92187c0418ed600d5bea27690fda40711122fdc28ce2f4

ARG ENABLE_1BIT_SGD

RUN test "$ENABLE_1BIT_SGD" = "true" ||  ( echo "\033[0;31m\
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
You are attempting to build an image of CNTK with 1-bit Stochastic Gradient\n\
Descent (1bit-SGD) enabled. Before proceeding further please ensure you     \n\ 
understand the license difference between CNTK and 1bit-SGD. See            \n\
https://docs.microsoft.com/en-us/cognitive-toolkit/Enabling-1bit-SGD#license-difference-between-cntk-and-1bit-sgd \n\
\n\
To build the image pass the following parameter to Docker:\n\
\n\
  --build-arg ENABLE_1BIT_SGD=true\n\
\n\
Example:\n\
\n\
  docker build -t cntk --build-arg ENABLE_1BIT_SGD=true ./CNTK-GPU-1bit-Image\n\
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
\n\033[0m"; exit 1)

RUN apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        build-essential \
        cmake \
        git \
        gfortran-multilib \
        libavcodec-dev \
        libavformat-dev \
        libjasper-dev \
        libjpeg-dev \
        libpng-dev \
        liblapacke-dev \
        libswscale-dev \
        libtiff-dev \
        pkg-config \
        wget \
        zlib1g-dev \
        # Protobuf
        ca-certificates \
        curl \
        unzip \
        # For Kaldi
        python-dev \
        automake \
        libtool-bin \
        autoconf \
        subversion \
        # For Kaldi's dependencies
        libapr1 libaprutil1 libltdl-dev libltdl7 libserf-1-1 libsigsegv2 libsvn1 m4 \
        # For Java Bindings
        openjdk-8-jdk \
        # For SWIG
        libpcre3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN OPENMPI_VERSION=1.10.3 && \
    wget -q -O - https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/mpi && \
    make -j"$(nproc)" install && \
    rm -rf /openmpi-${OPENMPI_VERSION}

ENV PATH /usr/local/mpi/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/mpi/lib:$LD_LIBRARY_PATH

RUN LIBZIP_VERSION=1.1.2 && \
    wget -q -O - https://libzip.org/download/libzip-${LIBZIP_VERSION}.tar.gz | tar -xzf - && \
    cd libzip-${LIBZIP_VERSION} && \
    ./configure && \
    make -j"$(nproc)" install && \
    rm -rf /libzip-${LIBZIP_VERSION}

ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

RUN wget -q -O - https://github.com/NVlabs/cub/archive/1.7.4.tar.gz | tar -C /usr/local -xzf -

RUN OPENCV_VERSION=3.1.0 && \
    wget -q -O - https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.tar.gz | tar -xzf - && \
    cd opencv-${OPENCV_VERSION} && \
    cmake -DWITH_CUDA=OFF -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-${OPENCV_VERSION} . && \
    make -j"$(nproc)" install && \
    rm -rf /opencv-${OPENCV_VERSION}

RUN OPENBLAS_VERSION=0.2.18 && \
    wget -q -O - https://github.com/xianyi/OpenBLAS/archive/v${OPENBLAS_VERSION}.tar.gz | tar -xzf - && \
    cd OpenBLAS-${OPENBLAS_VERSION} && \
    make -j"$(nproc)" USE_OPENMP=1 | tee make.log && \
    grep -qF 'OpenBLAS build complete. (BLAS CBLAS LAPACK LAPACKE)' make.log && \
    grep -qF 'Use OpenMP in the multithreading.' make.log && \
    make PREFIX=/usr/local/openblas install && \
    rm -rf /OpenBLAS-${OPENBLAS_VERSION}

ENV LD_LIBRARY_PATH /usr/local/openblas/lib:$LD_LIBRARY_PATH

# Install Boost
RUN BOOST_VERSION=1_60_0 && \
    BOOST_DOTTED_VERSION=$(echo $BOOST_VERSION | tr _ .) && \
    wget -q -O - https://sourceforge.net/projects/boost/files/boost/${BOOST_DOTTED_VERSION}/boost_${BOOST_VERSION}.tar.gz/download | tar -xzf - && \
    cd boost_${BOOST_VERSION} && \
    ./bootstrap.sh --prefix=/usr/local/boost-${BOOST_DOTTED_VERSION} --with-libraries=filesystem,system,test  && \
    ./b2 -d0 -j"$(nproc)" install  && \
    rm -rf /boost_${BOOST_VERSION}

# Install Protobuf
RUN PROTOBUF_VERSION=3.1.0 \
    PROTOBUF_STRING=protobuf-$PROTOBUF_VERSION && \
    wget -O - --no-verbose https://github.com/google/protobuf/archive/v${PROTOBUF_VERSION}.tar.gz | tar -xzf - && \
    cd $PROTOBUF_STRING && \
    ./autogen.sh && \
    ./configure CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared --prefix=/usr/local/$PROTOBUF_STRING && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf $PROTOBUF_STRING

# Install MKLDNN and MKLML
RUN mkdir /usr/local/mklml && \
    wget --no-verbose -O - https://github.com/01org/mkl-dnn/releases/download/v0.12/mklml_lnx_2018.0.1.20171227.tgz | \
    tar -xzf - -C /usr/local/mklml && \
    wget --no-verbose -O - https://github.com/01org/mkl-dnn/archive/v0.12.tar.gz | tar -xzf - && \
    cd mkl-dnn-0.12 && \
    ln -s /usr/local external && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    cd ../.. && \
    rm -rf mkl-dnn-0.12

# Install Kaldi
ENV KALDI_VERSION=c024e8aa
ENV KALDI_PATH /usr/local/kaldi-$KALDI_VERSION

RUN mv /bin/sh /bin/sh.orig && \
   ln -s -f /bin/bash /bin/sh && \
   mkdir $KALDI_PATH && \
   wget --no-verbose -O - https://github.com/kaldi-asr/kaldi/archive/$KALDI_VERSION.tar.gz | tar -xzf - --strip-components=1 -C $KALDI_PATH && \
   cd $KALDI_PATH && \
   cd tools && \
   perl -pi -e 's/^# (OPENFST_VERSION = 1.4.1)$/\1/' Makefile && \
   ./extras/check_dependencies.sh && \
   make -j $(nproc) all && \
   cd ../src && \
   # remove Fermi support as CUDA 9 no longer works on it
   perl -pi -e 's/-gencode arch=compute_20,code=sm_20//' cudamatrix/Makefile && \
   ./configure --openblas-root=/usr/local/openblas --shared && \
   make -j $(nproc) depend && \
   make -j $(nproc) all && \
# Remove some unneeded stuff in $KALDI_PATH to reduce size
   find $KALDI_PATH -name '*.o' -print0 | xargs -0 rm && \
   for dir in $KALDI_PATH/src/*bin; do make -C $dir clean; done && \
   mv -f /bin/sh.orig /bin/sh

## PYTHON

# Commit that will be used for Python environment creation (and later, compilation)
ARG COMMIT=master

# Swig
RUN cd /root && \
    wget -q http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz -O - | tar xvfz - && \
    cd swig-3.0.10 && \
    # Note: we specify --without-alllang to suppress building tests and examples for specific languages.
    ./configure --without-alllang && \
    make -j $(nproc) && \
    make install

# Anaconda
RUN wget -q https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh && \
    bash Anaconda3-4.2.0-Linux-x86_64.sh -b && \
    rm Anaconda3-4.2.0-Linux-x86_64.sh

RUN CONDA_ENV_PATH=/tmp/conda-linux-cntk-py35-environment.yml; \
    wget -q https://raw.githubusercontent.com/Microsoft/CNTK/$COMMIT/Scripts/install/linux/conda-linux-cntk-py35-environment.yml -O "$CONDA_ENV_PATH" && \
    /root/anaconda3/bin/conda env create -p /root/anaconda3/envs/cntk-py35 --file "$CONDA_ENV_PATH" && \
    rm -f "$CONDA_ENV_PATH"

ENV PATH /root/anaconda3/envs/cntk-py35/bin:$PATH

WORKDIR /cntk

# Allow CNTK's configure to pick up GDK and CuDNN where it expects it.
# (Note: $CUDNN_VERSION is defined by NVidia's base image)
# TODO hack, CNTK configure should be improved.
RUN _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2) && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/include && \
    ln -s /usr/include/cudnn.h /usr/local/cudnn-$_CUDNN_VERSION/cuda/include/cudnn.h && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64 && \
    ln -s /etc/alternatives/libcudnn_so /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64/libcudnn.so && \
    ln -s /usr/local/cudnn{-$_CUDNN_VERSION,} && \
    mkdir -p /usr/src/gdk/nvml/lib && \
    cp -av /usr/local/cuda/lib64/stubs/libnvidia-ml* /usr/src/gdk/nvml/lib && \
    cp -av /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/src/gdk/nvml/lib/libnvidia-ml.so.1 && \
    mkdir -p /usr/include/nvidia/gdk && \
    cp -av /usr/local/cuda/include/nvml.h /usr/include/nvidia/gdk/nvml.h

# Build CNTK
RUN git clone --depth=1 --recursive -b $COMMIT https://github.com/Microsoft/CNTK.git cntksrc && \
    cd cntksrc && \
    MKLML_VERSION_DETAIL=mklml_lnx_2018.0.1.20171227 && \
    CONFIGURE_OPTS="\
      --1bitsgd=yes \
      --with-kaldi=${KALDI_PATH} \
      --with-py35-path=/root/anaconda3/envs/cntk-py35" && \
    mkdir -p build/gpu/release && \
    cd build/gpu/release && \
    ../../../configure $CONFIGURE_OPTS --with-openblas=/usr/local/openblas && \
    make -j"$(nproc)" all && \
    cd ../../.. && \
    mkdir -p build-mkl/gpu/release && \
    cd build-mkl/gpu/release && \
    ../../../configure $CONFIGURE_OPTS --with-mkl=/usr/local/mklml/$MKLML_VERSION_DETAIL && \
    make -j"$(nproc)" all

RUN cd cntksrc/Examples/Image/DataSets/CIFAR-10 && \
    python install_cifar10.py && \
    cd ../../../..

RUN cd cntksrc/Examples/Image/DataSets/MNIST && \
    python install_mnist.py && \
    cd ../../../..

ENV PATH=/cntk/cntksrc/build/gpu/release/bin:$PATH PYTHONPATH=/cntk/cntksrc/bindings/python LD_LIBRARY_PATH=/cntk/cntksrc/bindings/python/cntk/libs:$LD_LIBRARY_PATH

# Install CNTK as the default backend for Keras
ENV KERAS_BACKEND=cntk