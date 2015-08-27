import sys
import urllib
import tarfile
import shutil
import os
import struct
import numpy as np
import cPickle as cp

ImgSize = 32
NumFeat = ImgSize * ImgSize * 3

def readBatch(src):
    with open(src, 'rb') as f:
        d = cp.load(f)
        # Note: most of the frameworks use convolution-friendly input format (color channel major):
        # R0..RN,G0..GN,B0..BN
        # while CNTK uses 'RGB' like format.
        # CIFAR-10 dataset comes in convolution-friendly format so it has to be converted to CNTK format.
        data = d['data']
        r = data[:, : ImgSize * ImgSize]
        g = data[:, ImgSize * ImgSize : 2 * ImgSize * ImgSize]
        b = data[:, 2 * ImgSize * ImgSize : 3 * ImgSize * ImgSize]
        feat = np.empty_like(data)
        feat[:, ::3] = r
        feat[:, 1::3] = g
        feat[:, 2::3] = b
    return np.hstack((np.reshape(d['labels'], (len(d['labels']), 1)), feat))

def loadData(src):
    print 'Downloading ' + src
    fname, h = urllib.urlretrieve(src, './delete.me')
    print 'Done.'
    try:
        print 'Extracting files...'
        with tarfile.open(fname) as tar:
            tar.extractall()
        print 'Done.'
        print 'Preparing train set...'
        trn = np.empty((0, NumFeat + 1))
        for i in range(5):
            batchName = './cifar-10-batches-py/data_batch_{0}'.format(i + 1)
            trn = np.vstack((trn, readBatch(batchName)))
        print 'Done.'
        print 'Preparing test set...'
        tst = readBatch('./cifar-10-batches-py/test_batch')
        print 'Done.'
    finally:
        os.remove(fname)
    return (trn, tst)

if __name__ == "__main__":
    trn, tst = loadData('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    print 'Writing train text file...'
    np.savetxt(r'./Train.txt', trn, fmt = '%u', delimiter='\t')
    print 'Done.'
    print 'Writing test text file...'
    np.savetxt(r'./Test.txt', tst, fmt = '%u', delimiter='\t')
    print 'Done.'
