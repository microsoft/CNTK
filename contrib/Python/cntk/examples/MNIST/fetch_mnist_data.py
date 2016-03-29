import urllib
import gzip
import os
import struct
import numpy as np


def loadData(src, cimg):
    print('Downloading ' + src)
    gzfname, h = urllib.request.urlretrieve(src, './delete.me')
    print('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception(
                    'Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception(
                    'Invalid file: expected 28 rows/cols per image.')
            # Read data.
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype=np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))


def loadLabels(src, cimg):
    print('Downloading ' + src)
    gzfname, h = urllib.request.urlretrieve(src, './delete.me')
    print('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception(
                    'Invalid file: expected {0} rows.'.format(cimg))
            # Read labels.
            res = np.fromstring(gz.read(cimg), dtype=np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, 1))


if __name__ == "__main__":
    trnData = loadData(
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 60000)
    trnLbl = loadLabels(
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 60000)
    trn = np.hstack((trnLbl, trnData))
    if not os.path.exists('./Data'):
        os.mkdir('./Data')
    print('Writing train text file...')
    np.savetxt(r'./Data/Train-28x28.txt', trn, fmt='%u', delimiter='\t')
    print('Done.')
    testData = loadData(
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 10000)
    testLbl = loadLabels(
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 10000)
    test = np.hstack((testLbl, testData))
    print('Writing test text file...')
    np.savetxt(r'./Data/Test-28x28.txt', test, fmt='%u', delimiter='\t')
    print('Done.')
