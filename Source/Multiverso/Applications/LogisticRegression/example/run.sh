cd ../../../
mkdir build
cd build
cmake .. && make

cd ../Applications/LogisticRegression/example/

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz && gunzip train-images-idx3-ubyte.gz &
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz && gunzip train-labels-idx1-ubyte.gz &
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz && gunzip t10k-images-idx3-ubyte.gz &
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz && gunzip t10k-labels-idx1-ubyte.gz &
wait

python convert.py train && rm train-images-idx3-ubyte -f && rm train-labels-idx1-ubyte -f &
python convert.py test && rm t10k-images-idx3-ubyte -f && rm t10k-labels-idx1-ubyte -f &
wait

../../../build/Applications/LogisticRegression/LogisticRegression mnist.config
