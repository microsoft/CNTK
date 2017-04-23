FROM ubuntu:14.04
MAINTAINER lyysdy@foxmail.com

USER root

# install dev tools
RUN apt-get update \
    && apt-get install -qqy \
       curl \
       tar \
       g++-4.8 \
       gcc \
       libtool \
       pkg-config \
       autoconf \
       openssh-server \
       openssh-client \
       rsync \
       build-essential \
       automake \
       vim \
       gdb \
       git \
       libopenmpi-dev \
       openmpi-bin \
       cmake \
       gfortran \
       python-nose \
       python-numpy \
       python-scipy \
       python-dev \
       python-pip \
       libopenblas-dev \
       software-properties-common \
       libssl-dev \
       libzmq3-dev \
       python-zmq 

# java
RUN mkdir -p /usr/local/java/default && \
    curl -Ls 'http://download.oracle.com/otn-pub/java/jdk/8u65-b17/jdk-8u65-linux-x64.tar.gz' -H 'Cookie: oraclelicense=accept-securebackup-cookie' | \
    tar --strip-components=1 -xz -C /usr/local/java/default/

ENV JAVA_HOME /usr/local/java/default/ 
ENV PATH $PATH:$JAVA_HOME/bin

# hadoop
RUN wget -cq -t 0 http://www.eu.apache.org/dist/hadoop/common/hadoop-2.6.0/hadoop-2.6.0.tar.gz 
RUN tar -xz -C /usr/local/ -f hadoop-2.6.0.tar.gz \
    && rm hadoop-2.6.0.tar.gz \
    && cd /usr/local && ln -s ./hadoop-2.6.0 hadoop \
    && cp -r /usr/local/hadoop/include/* /usr/local/include

ENV HADOOP_PREFIX /usr/local/hadoop
RUN sed -i '/^export JAVA_HOME/ s:.*:export JAVA_HOME=/usr/local/java/default\nexport HADOOP_PREFIX=/usr/local/hadoop\nexport HADOOP_HOME=/usr/local/hadoop\n:' $HADOOP_PREFIX/etc/hadoop/hadoop-env.sh
RUN sed -i '/^export HADOOP_CONF_DIR/ s:.*:export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop/:' $HADOOP_PREFIX/etc/hadoop/hadoop-env.sh

# fixing the libhadoop.so like a boss
RUN rm  /usr/local/hadoop/lib/native/* \
    && curl -Ls http://dl.bintray.com/sequenceiq/sequenceiq-bin/hadoop-native-64-2.6.0.tar | tar -x -C /usr/local/hadoop/lib/native/

# install Theano-dev
RUN mkdir -p /theano \
    && cd /theano \
    && git clone git://github.com/Theano/Theano.git \
    && cd /theano/Theano \
    && python setup.py develop

# Install Jupyter Notebook for iTorch
RUN pip install notebook ipywidgets

# Run Torch7 installation scripts
RUN git clone https://github.com/torch/distro.git /root/torch --recursive && cd /root/torch && \
  bash install-deps && \
  ./install.sh


# Export environment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

WORKDIR /dmtk
   
RUN cd /dmtk && git clone https://github.com/Microsoft/multiverso.git && cd multiverso \
	&& mkdir build && cd build \
	&& cmake .. && make && make install 

# python tests
RUN cd /dmtk/multiverso/binding/python \
	&& python setup.py install \
	&& nosetests

# lua tests
RUN cd /dmtk/multiverso/binding/lua \
	&& make install \
	&& make test

# run cpp tests
RUN cd /dmtk/multiverso/build \
   && mpirun -np 4 ./Test/multiverso.test kv \
   && mpirun -np 4 ./Test/multiverso.test array \
   && mpirun -np 4 ./Test/multiverso.test net \
   && mpirun -np 4 ./Test/multiverso.test ip \
   && mpirun -np 4 ./Test/multiverso.test checkpoint \
   && mpirun -np 4 ./Test/multiverso.test restore \
   && mpirun -np 4 ./Test/multiverso.test allreduce
# - mpirun -np 4 ./Test/multiverso.test matrix  # TODO the matrix test won't stop
# - mpirun -np 4 ./Test/multiverso.test TestSparsePerf # TODO TestSparsePerf takes too much time
# - mpirun -np 4 ./Test/multiverso.test TestDensePerf # TODO TestDensePerf takes too much time

# clean unnessary packages
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

