# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================


import urllib
import gzip
import os
import struct
import numpy as np

# This is a script to download and prepare MNIST training and testing data

def load_data(src, cimg):
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


def load_labels(src, cimg):
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
    taining_data = load_data(
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 60000)
    training_labels = load_labels(
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 60000)
    trn = np.hstack((training_labels, taining_data))
    if not os.path.exists('./Data'):
        os.mkdir('./Data')
    print('Writing train text file...')
    np.savetxt(r'./Data/Train-28x28.txt', trn, fmt='%u', delimiter='\t')
    print('Done.')
    testing_data = load_data(
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 10000)
    testing_labels = load_labels(
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 10000)
    test = np.hstack((testing_labels, testing_data))
    print('Writing test text file...')
    np.savetxt(r'./Data/Test-28x28.txt', test, fmt='%u', delimiter='\t')
    print('Done.')
