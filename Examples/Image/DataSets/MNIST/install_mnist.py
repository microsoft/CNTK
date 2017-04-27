from __future__ import print_function
import os
import mnist_utils as ut

if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    train = ut.load('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 60000)
    print ('Writing train text file...')
    ut.savetxt(r'./Train-28x28_cntk_text.txt', train)
    print ('Done.')
    test = ut.load('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 10000)
    print ('Writing test text file...')
    ut.savetxt(r'./Test-28x28_cntk_text.txt', test)
    print ('Done.')
