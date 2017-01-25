FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

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
        libtool \
        autoconf \
        subversion \
        # For Kaldi's dependencies
        libapr1 libaprutil1 libltdl-dev libltdl7 libserf-1-1 libsigsegv2 libsvn1 m4 \
        # For SWIG
        libpcre++-dev && \
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
    wget -q -O - http://nih.at/libzip/libzip-${LIBZIP_VERSION}.tar.gz | tar -xzf - && \
    cd libzip-${LIBZIP_VERSION} && \
    ./configure && \
    make -j"$(nproc)" install && \
    rm -rf /libzip-${LIBZIP_VERSION}

ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

RUN wget -q -O - https://github.com/NVlabs/cub/archive/1.4.1.tar.gz | tar -C /usr/local -xzf -

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

# Install CNTK custom MKL
RUN CNTK_CUSTOM_MKL_VERSION=3 && \
    mkdir /usr/local/CNTKCustomMKL && \
    wget --no-verbose -O - https://www.cntk.ai/mkl/CNTKCustomMKL-Linux-$CNTK_CUSTOM_MKL_VERSION.tgz | \
    tar -xzf - -C /usr/local/CNTKCustomMKL

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
   ./configure --openblas-root=/usr/local/openblas --shared && \
   make -j $(nproc) depend && \
   make -j $(nproc) all && \
# Remove some unneeded stuff in $KALDI_PATH to reduce size
   find $KALDI_PATH -name '*.o' -print0 | xargs -0 rm && \
   for dir in $KALDI_PATH/src/*bin; do make -C $dir clean; done && \
   mv -f /bin/sh.orig /bin/sh

## PYTHON

# Swig
RUN cd /root && \
    wget -q http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz -O - | tar xvfz - && \
    cd swig-3.0.10 && \
    ./configure --without-java --without-perl5 && \
    make -j $(nproc) && \
    make install

# Anaconda
RUN wget -q https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh && \
    bash Anaconda3-4.2.0-Linux-x86_64.sh -b && \
    rm Anaconda3-4.2.0-Linux-x86_64.sh

RUN CONDA_ENV_PATH=/tmp/conda-linux-cntk-py35-environment.yml; \
    wget -q https://raw.githubusercontent.com/Microsoft/CNTK/master/Scripts/install/linux/conda-linux-cntk-py35-environment.yml -O "$CONDA_ENV_PATH" && \
    /root/anaconda3/bin/conda env create -p /root/anaconda3/envs/cntk-py35 --file "$CONDA_ENV_PATH" && \
    rm -f "$CONDA_ENV_PATH"

ENV PATH /root/anaconda3/envs/cntk-py35/bin:$PATH

# NCCL
ENV NCCL_VERSION=1.3.0-1
ENV NCCL_PATH /usr/local/nccl-${NCCL_VERSION}
RUN git clone --depth=1 -b v${NCCL_VERSION} https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make CUDA_HOME=/usr/local/cuda PREFIX=$NCCL_PATH install && \
    rm -rf /nccl

WORKDIR /cntk

RUN mkdir -p /usr/local/cudnn/cuda/include && \
    ln -s /usr/include/cudnn.h /usr/local/cudnn/cuda/include/cudnn.h && \
    mkdir -p /usr/local/cudnn/cuda/lib64 && \
    ln -s /etc/alternatives/libcudnn_so /usr/local/cudnn/cuda/lib64/libcudnn.so && \
    ln -s /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/local/cuda/lib64/stubs/libnvidia-ml.so.1

# Build CNTK
RUN git clone --depth=1 -b master https://github.com/Microsoft/CNTK.git . && \
    CONFIGURE_OPTS="\
      --with-cuda=/usr/local/cuda \
      --with-gdk-include=/usr/local/cuda/include \
      --with-gdk-nvml-lib=/usr/local/cuda/lib64/stubs \
      --with-kaldi=${KALDI_PATH} \
      --with-py35-path=/root/anaconda3/envs/cntk-py35 \
      --with-cudnn=/usr/local/cudnn \
      --with-nccl=${NCCL_PATH}" && \
    git submodule update --init Source/Multiverso && \
    mkdir -p build/gpu/release && \
    cd build/gpu/release && \
    ../../../configure $CONFIGURE_OPTS --with-openblas=/usr/local/openblas && \
    make -j"$(nproc)" all && \
    cd ../../.. && \
    mkdir -p build-mkl/gpu/release && \
    cd build-mkl/gpu/release && \
    ../../../configure $CONFIGURE_OPTS --with-mkl=/usr/local/CNTKCustomMKL && \
    make -j"$(nproc)" all

RUN cd Examples/Image/DataSets/CIFAR-10 && \
    python install_cifar10.py && \
    cd ../../../..

RUN cd Examples/Image/DataSets/MNIST && \
    python install_mnist.py && \
    cd ../../../..

ENV PATH=/cntk/build/gpu/release/bin:$PATH PYTHONPATH=/cntk/bindings/python LD_LIBRARY_PATH=/cntk/bindings/python/cntk/libs:$LD_LIBRARY_PATH
