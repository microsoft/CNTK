#!/bin/bash

wget http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz

tar xvfz swig-3.0.10.tar.gz

pushd swig-3.0.10

./configure --without-java --without-perl5 --prefix=$(readlink -m ./root)

make -j 4

make install
