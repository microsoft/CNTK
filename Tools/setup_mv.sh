
BUILD_TOP=$1

cd Source/Multiverso/build

cmake -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBoost_NO_SYSTEM_PATHS=TRUE \
      -DBOOST_ROOT:PATHNAME=/usr/local/boost-1.60.0 \
      -DBOOST_LIBRARY_DIRS:FILEPATH=/usr/local/boost-1.60.0/lib \
      -DLIBRARY_OUTPUT_PATH=$BUILD_TOP/lib \
      -DEXECUTABLE_OUTPUT_PATH=$BUILD_TOP/bin \
      ..

make -j multiverso
make -j multiversotests
