# CNTK Dockerfile
#   CPU only
#   No 1-bit SGD

FROM ubuntu:14.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        build-essential \
        cmake \
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
        python-dev \
        python-numpy \
        python-pip \
        python-yaml \
        wget \
        zlib1g-dev \
        #For Kaldi
        automake \
        libtool \
        autoconf \
        subversion \
        #For Kaldi's dependencies
        libapr1 libaprutil1 libltdl-dev libltdl7 libserf-1-1 libsigsegv2 libsvn1 m4 && \
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

# Install CNTK custom MKL, version 2
RUN mkdir /usr/local/CNTKCustomMKL && \
    wget --no-verbose -O - https://www.cntk.ai/mkl/CNTKCustomMKL-Linux-2.tgz | \
    tar -xzf - -C /usr/local/CNTKCustomMKL

RUN pip install pytest sphinx-rtd-theme Pillow

ENV BLAS=/usr/local/openblas/lib/libopenblas.so
ENV LAPACK=/usr/local/openblas/lib/libopenblas.so

RUN SCIPY_VERSION=0.18.0 && \
    wget -q -O - https://github.com/scipy/scipy/releases/download/v${SCIPY_VERSION}/scipy-${SCIPY_VERSION}.tar.gz | tar -zxf - && \
    cd scipy-${SCIPY_VERSION} && \
    python setup.py build && \
    python setup.py install && \
    cd .. && \
    rm -rf scipy-${SCIPY_VERSION}

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

WORKDIR /cntk

RUN git clone --depth=1 -b master https://github.com/Microsoft/CNTK.git . && \
    mkdir -p build/cpu/release && \
    cd build/cpu/release && \
    ../../../configure --with-openblas=/usr/local/openblas --with-kaldi=${KALDI_PATH} && \
    make -j"$(nproc)" all && \
    cd ../../.. && \
    mkdir -p build-mkl/cpu/release && \
    cd build-mkl/cpu/release && \
    ../../../configure --with-mkl=/usr/local/CNTKCustomMKL --with-kaldi=${KALDI_PATH} && \
    make -j"$(nproc)" all

RUN cd Examples/Image/Miscellaneous/CIFAR-10 && \
    python CifarDownload.py && \
    python CifarConverter.py cifar-10-batches-py && \
    cd ../../../..

RUN cd Examples/Image/MNIST/AdditionalFiles && \
    python mnist_convert.py && \
    cd ../../../..

ENV PATH=/cntk/build/cpu/release/bin:$PATH
