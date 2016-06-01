import sys
import urllib
import tarfile
import shutil
import os
import struct
import numpy as np
import cPickle as cp
import getopt

ImgSize = 32
NumFeat = ImgSize * ImgSize * 3

sys.path.append("../../../../Source/Readers/CNTKTextFormatReader")
from uci_to_cntk_text_format_converter import convert

def readBatch(src, outFmt):
    with open(src, 'rb') as f:
        d = cp.load(f)
        # Note: most of the frameworks use spatial-major (aka NCHW) input format:
        # R0..RN,G0..GN,B0..BN
        # There are 2 possible options in CNTK:
        # 1. If CNTK is built with cuDNN then 'cudnn' (i.e. NCHW format) should be used.
        # 2. Otherwise, legacy CNTK 'NHWC' format should be used. As CIFAR-10 dataset comes in 
        #   NCHW format, it has to be converted to CNTK legacy format first.
        data = d['data']
        if outFmt == 'cudnn':
            feat = data
        elif outFmt == 'legacy':
            r = data[:, : ImgSize * ImgSize]
            g = data[:, ImgSize * ImgSize : 2 * ImgSize * ImgSize]
            b = data[:, 2 * ImgSize * ImgSize : 3 * ImgSize * ImgSize]
            feat = np.empty_like(data)
            feat[:, ::3] = r
            feat[:, 1::3] = g
            feat[:, 2::3] = b
        else:
            print ('Format not supported: ' + outFmt)
            usage()
            sys.exit(1)
    return np.hstack((np.reshape(d['labels'], (len(d['labels']), 1)), feat))

def loadData(src, outFmt):
    print ('Downloading ' + src)
    fname, h = urllib.urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        print ('Extracting files...')
        with tarfile.open(fname) as tar:
            tar.extractall()
        print ('Done.')
        print ('Preparing train set...')
        trn = np.empty((0, NumFeat + 1))
        for i in range(5):
            batchName = './cifar-10-batches-py/data_batch_{0}'.format(i + 1)
            trn = np.vstack((trn, readBatch(batchName, outFmt)))
        print ('Done.')
        print ('Preparing test set...')
        tst = readBatch('./cifar-10-batches-py/test_batch', outFmt)
        print ('Done.')
    finally:
        os.remove(fname)
    return (trn, tst)

def usage():
    print ('Usage: CIFAR_convert.py [-f <format>] \n  where format can be either cudnn or legacy. Default is cudnn.')

def parseCmdOpt(argv):
    if len(argv) == 0:
        print ("Using cudnn output format.")
        return "cudnn"
    try:
        opts, args = getopt.getopt(argv, 'hf:', ['help', 'outFormat='])
    except getopt.GetoptError:
        usage()
        sys.exit(1)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt in ('-f', '--outFormat'):
            fmt = arg
            if fmt != 'cudnn' and fmt != 'legacy':
                print ('Invalid output format option.')
                usage()
                sys.exit(1)
            return fmt

if __name__ == "__main__":
    fmt = parseCmdOpt(sys.argv[1:])
    trn, tst = loadData('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', fmt)
    print ('Writing train text file...')
    np.savetxt(r'./Train.txt', trn, fmt = '%u', delimiter='\t')
    convert(r'./Train.txt', r'./Train_cntk_text.txt', 1, 3072, 0, 1, 10)
    os.remove(r'./Train.txt');
    print ('Done.')
    print ('Writing test text file...')
    np.savetxt(r'./Test.txt', tst, fmt = '%u', delimiter='\t')
    convert(r'./Test.txt', r'./Test_cntk_text.txt', 1, 3072, 0, 1, 10)
    os.remove(r'./Test.txt');
    print ('Done.')
