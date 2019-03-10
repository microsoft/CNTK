# CNTK Dockerfile
#   CPU only
#   No 1-bit SGD
#
# To build, run from the parent with the command line:
# 	docker build -t <image name> -f CNTK-CPUOnly-Image/Dockerfile .

FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        build-essential \
        git \
        g++-multilib \
        gcc-multilib \
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
        # OPENSSL support
        libssl-dev \
        openssl \
        # Protobuf
        ca-certificates \
        curl \
		libcurl4-openssl-dev \
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
        libpcre3-dev \
        # For graphics managed lib
        libgdiplus \
        # .NET Core SDK
        apt-transport-https && \
        # Cleanup
        rm -rf /var/lib/apt/lists/*

ARG CMAKE_DOWNLOAD_VERSION=3.11
ARG CMAKE_BUILD_VERSION=4
RUN DEBIAN_FRONTEND=noninteractive && \
    wget --no-verbose https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/cmake/$CMAKE_DOWNLOAD_VERSION/cmake-$CMAKE_DOWNLOAD_VERSION.$CMAKE_BUILD_VERSION.tar.gz && \
    tar -xzvf cmake-$CMAKE_DOWNLOAD_VERSION.$CMAKE_BUILD_VERSION.tar.gz && \
    cd cmake-$CMAKE_DOWNLOAD_VERSION.$CMAKE_BUILD_VERSION && \
    ./bootstrap --system-curl -- -DCMAKE_USE_OPENSSL=ON && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf cmake-$CMAKE_DOWNLOAD_VERSION.$CMAKE_BUILD_VERSION	

ARG OPENMPI_VERSION=1.10.7
RUN wget -q -O - https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/openmpi/$OPENMPI_VERSION/openmpi-$OPENMPI_VERSION.tar.gz | tar -xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    apt-get -y update && \
    apt-get -y -f install && \
    apt-get -y install libsysfs2 libsysfs-dev && \
    ./configure --with-verbs --with-cuda=/usr/local/cuda --prefix=/usr/local/mpi && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf openmpi-${OPENMPI_VERSION}
ENV PATH /usr/local/mpi/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/mpi/lib:$LD_LIBRARY_PATH

ARG LIBZIP_VERSION=1.1.2
RUN wget -q -O - https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/libzip/$LIBZIP_VERSION/libzip-$LIBZIP_VERSION.tar.gz | tar -xzf - && \
    cd libzip-${LIBZIP_VERSION} && \
    ./configure && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf libzip-${LIBZIP_VERSION}
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

ARG OPENCV_VERSION=3.1.0
RUN wget -q -O - https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/opencv/$OPENCV_VERSION/opencv-$OPENCV_VERSION.tar.gz | tar -xzf - && \
    cd opencv-${OPENCV_VERSION} && \
    cmake -DWITH_CUDA=OFF -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-${OPENCV_VERSION} . && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf opencv-${OPENCV_VERSION}

ARG OPENBLAS_VERSION=0.2.18
RUN wget -q -O - https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/openblas/$OPENBLAS_VERSION/OpenBLAS-$OPENBLAS_VERSION.tar.gz | tar -xzf - && \
    cd OpenBLAS-${OPENBLAS_VERSION} && \
    make -j 2 MAKE_NB_JOBS=0 USE_OPENMP=1 | tee make.log && \
    grep -qF 'OpenBLAS build complete. (BLAS CBLAS LAPACK LAPACKE)' make.log && \
    grep -qF 'Use OpenMP in the multithreading.' make.log && \
    make PREFIX=/usr/local/openblas install && \
    cd .. && \
    rm -rf OpenBLAS-${OPENBLAS_VERSION}
ENV LD_LIBRARY_PATH /usr/local/openblas/lib:$LD_LIBRARY_PATH

# Install Boost
ARG BOOST_VERSION=1.60.0
RUN BOOST_UNDERSCORE_VERSION=$(echo $BOOST_VERSION | tr . _) && \
    wget -q -O - https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/boost/$BOOST_VERSION/boost_$BOOST_UNDERSCORE_VERSION.tar.gz | tar -xzf - && \
    cd boost_${BOOST_UNDERSCORE_VERSION} && \
    ./bootstrap.sh --prefix=/usr/local/boost-${BOOST_VERSION}  && \
    ./b2 -d0 -j $(nproc) install && \
    cd .. && \
    rm -rf boost_${BOOST_UNDERSCORE_VERSION}

# Install Protobuf
ARG PROTOBUF_VERSION=3.1.0
RUN PROTOBUF_STRING=protobuf-$PROTOBUF_VERSION && \
    wget -O - --no-verbose https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/protobuf/$PROTOBUF_VERSION/protobuf-$PROTOBUF_VERSION.tar.gz | tar -xzf - && \
    cd $PROTOBUF_STRING && \
    ./autogen.sh && \
    ./configure CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared --prefix=/usr/local/$PROTOBUF_STRING && \
    make -j $(nproc) install && \
    cd .. && \
    rm -rf $PROTOBUF_STRING

# Install MKLDNN and MKLML
ARG MKLDNN_VERSION=0.14
ARG MKLDNN_LONG_VERSION=mklml_lnx_2018.0.3.20180406
RUN mkdir /usr/local/mklml && \
    wget --no-verbose -O - https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/mkl-dnn/$MKLDNN_VERSION/$MKLDNN_LONG_VERSION.tgz | \
    tar -xzf - -C /usr/local/mklml && \
    MKLDNN_STRING=mkl-dnn-${MKLDNN_VERSION} && \
    wget --no-verbose -O - https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/mkl-dnn/$MKLDNN_VERSION/mkl-dnn-$MKLDNN_VERSION.tar.gz | tar -xzf - && \
    cd ${MKLDNN_STRING} && \
    ln -s /usr/local external && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/ && \
    make && \
    make install DESTDIR=/usr/local && \
    make install DESTDIR=/usr/local/mklml/${MKLDNN_LONG_VERSION} && \
    cd ../.. && \
    rm -rf ${MKLDNN_STRING}

# Install Kaldi
ARG KALDI_VERSION=c024e8aa
ARG KALDI_PATH=/usr/local/kaldi-$KALDI_VERSION
RUN mv /bin/sh /bin/sh.orig && \
    ln -s -f /bin/bash /bin/sh && \
    mkdir $KALDI_PATH && \
    wget --no-verbose -O - https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/kaldi/$KALDI_VERSION/kaldi-$KALDI_VERSION.tar.gz | tar -xzf - --strip-components=1 -C $KALDI_PATH && \
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
ARG SWIG_VERSION=3.0.10
ARG CACHEBUST=1
RUN wget -q https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/swig/$SWIG_VERSION/swig-$SWIG_VERSION.tar.gz -O - | tar xvfz - && \
    cd swig-$SWIG_VERSION && \
    # Note: we specify --without-alllang to suppress building tests and examples for specific languages.
    ./configure --prefix=/usr/local/swig-$SWIG_VERSION --without-perl5 --without-alllang && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -rf swig-$SWIG_VERSION
COPY ./Patches /tmp/patches
RUN /tmp/patches/patch_swig.sh /usr/local/share/swig/3.0.10 && \
	rm -rfd /tmp/patches

# .NET Core SDK
RUN wget -q https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/packages-microsoft-prod/deb/packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get -y -f install apt-transport-https && \
    apt-get -y update && \
    apt-get -y -f install dotnet-sdk-2.1 && \
    rm ./packages-microsoft-prod.deb

# Anaconda
ARG ANACONDA_VERSION=4.2.0
RUN wget -q https://cntkbuildstorage.blob.core.windows.net/cntk-ci-dependencies/anaconda3/$ANACONDA_VERSION/Anaconda3-$ANACONDA_VERSION-Linux-x86_64.sh && \
    bash Anaconda3-$ANACONDA_VERSION-Linux-x86_64.sh -b && \
    rm Anaconda3-$ANACONDA_VERSION-Linux-x86_64.sh

RUN CONDA_ENV_PATH=/tmp/conda-linux-cntk-py35-environment.yml; \
    wget -q https://raw.githubusercontent.com/Microsoft/CNTK/$COMMIT/Scripts/install/linux/conda-linux-cntk-py35-environment.yml -O "$CONDA_ENV_PATH" && \
    /root/anaconda3/bin/conda env create -p /root/anaconda3/envs/cntk-py35 --file "$CONDA_ENV_PATH" && \
    rm -f "$CONDA_ENV_PATH"

ENV PATH /root/anaconda3/envs/cntk-py35/bin:$PATH

WORKDIR /cntk

# Build CNTK
RUN git clone --depth=1 --recursive -b $COMMIT https://github.com/Microsoft/CNTK.git cntksrc && \
    cd cntksrc && \
    MKLML_VERSION_DETAIL=${MKLDNN_LONG_VERSION} && \
    CONFIGURE_OPTS="\
      --with-kaldi=${KALDI_PATH} \
      --with-py35-path=/root/anaconda3/envs/cntk-py35" && \
    mkdir -p build/cpu/release && \
    cd build/cpu/release && \
    ../../../configure $CONFIGURE_OPTS --with-openblas=/usr/local/openblas && \
    make -j"$(nproc)" all && \
    cd ../../.. && \
    mkdir -p build-mkl/cpu/release && \
    cd build-mkl/cpu/release && \
    ../../../configure $CONFIGURE_OPTS --with-mkl=/usr/local/mklml/${MKLML_VERSION_DETAIL} && \
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
