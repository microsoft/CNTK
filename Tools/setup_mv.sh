cd Source/Multiverso/build

cmake -DBoost_NO_BOOST_CMAKE=TRUE \
      -DBoost_NO_SYSTEM_PATHS=TRUE \
      -DBOOST_ROOT:PATHNAME=/usr/local/boost-1.60.0 \
      -DBOOST_LIBRARY_DIRS:FILEPATH=/usr/local/boost-1.60.0/lib \
      ..

make -j multiverso
make -j multiverso.ut
