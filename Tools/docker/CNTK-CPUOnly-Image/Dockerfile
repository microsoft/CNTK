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
    wget --no-verbose https://cmake.org/files/v$CMAKE_DOWNLOAD_VERSION/cmake-$CMAKE_DOWNLOAD_VERSION.$CMAKE_BUILD_VERSION.tar.gz && \
	tar -xzvf cmake-$CMAKE_DOWNLOAD_VERSION.$CMAKE_BUILD_VERSION.tar.gz && \
    cd cmake-$CMAKE_DOWNLOAD_VERSION.$CMAKE_BUILD_VERSION && \
    ./bootstrap --system-curl -- -DCMAKE_USE_OPENSSL=ON && \
    make -j $(nproc) install && \
	cd .. && \
    rm -rf cmake-$CMAKE_DOWNLOAD_VERSION.$CMAKE_BUILD_VERSION	

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

RUN OPENCV_VERSION=3.1.0 && \
    wget -q -O - https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.tar.gz | tar -xzf - && \
    cd opencv-${OPENCV_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-${OPENCV_VERSION} . && \
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
ARG MKLDNN_VERSION=0.14
ARG MKLDNN_LONG_VERSION=mklml_lnx_2018.0.3.20180406
RUN mkdir /usr/local/mklml && \
    wget --no-verbose -O - https://github.com/01org/mkl-dnn/releases/download/v${MKLDNN_VERSION}/${MKLDNN_LONG_VERSION}.tgz | \
    tar -xzf - -C /usr/local/mklml && \
    MKLDNN_STRING=mkl-dnn-${MKLDNN_VERSION} && \
    wget --no-verbose -O - https://github.com/01org/mkl-dnn/archive/v${MKLDNN_VERSION}.tar.gz | tar -xzf - && \
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

COPY ./Patches /tmp/patches
RUN /tmp/patches/patch_swig.sh /usr/local/share/swig/3.0.10 && \
	rm -rfd /tmp/patches
	
# .NET Core SDK
RUN cd /tmp && \
    (wget -q packages-microsoft-prod.deb https://packages.microsoft.com/config/ubuntu/16.04/packages-microsoft-prod.deb || true) && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get --yes install dotnet-sdk-2.1.200 && \
    rm /tmp/packages-microsoft-prod.deb && \
    apt-get --yes autoremove && \
    rm -rf /var/lib/apt/lists/*

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
