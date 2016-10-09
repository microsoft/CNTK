from __future__ import print_function
try: 
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
import sys
import tarfile
import shutil
import os
import struct
import numpy as np
import pickle as cp
from PIL import Image
import xml.etree.cElementTree as et
import xml.dom.minidom
import getopt

ImgSize = 32
NumFeat = ImgSize * ImgSize * 3

def readBatch(src):
    with open(src, 'rb') as f:
        if sys.version_info[0] < 3: 
            d = cp.load(f) 
        else:
            d = cp.load(f, encoding='latin1')
        data = d['data']
        feat = data
    res = np.hstack((feat, np.reshape(d['labels'], (len(d['labels']), 1))))
    return res.astype(np.int)

def loadData(src):
    print ('Downloading ' + src)
    fname, h = urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        print ('Extracting files...')
        with tarfile.open(fname) as tar:
            tar.extractall()
        print ('Done.')
        print ('Preparing train set...')
        trn = np.empty((0, NumFeat + 1), dtype=np.int)
        for i in range(5):
            batchName = './cifar-10-batches-py/data_batch_{0}'.format(i + 1)
            trn = np.vstack((trn, readBatch(batchName)))
        print ('Done.')
        print ('Preparing test set...')
        tst = readBatch('./cifar-10-batches-py/test_batch')
        print ('Done.')
    finally:
        os.remove(fname)
    return (trn, tst)

def saveTxt(filename, ndarray):
    with open(filename, 'w') as f:
        labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
        for row in ndarray:
            row_str = row.astype(str)
            label_str = labels[row[-1]]
            feature_str = ' '.join(row_str[:-1])
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))

def saveImage(fname, data, label, mapFile, regrFile, pad, **key_parms):
    # data in CIFAR-10 dataset is in CHW format.
    pixData = data.reshape((3, ImgSize, ImgSize))
    if ('mean' in key_parms):
        key_parms['mean'] += pixData

    if pad > 0:
        pixData = np.pad(pixData, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=128) # can also use mode='edge'

    img = Image.new('RGB', (ImgSize + 2 * pad, ImgSize + 2 * pad))
    pixels = img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            pixels[x, y] = (pixData[0][y][x], pixData[1][y][x], pixData[2][y][x])
    img.save(fname)
    mapFile.write("%s\t%d\n" % (fname, label))
    
    # compute per channel mean and store for regression example
    channelMean = np.mean(pixData, axis=(1,2))
    regrFile.write("|regrLabels\t%f\t%f\t%f\n" % (channelMean[0]/255.0, channelMean[1]/255.0, channelMean[2]/255.0))
    
def saveMean(fname, data):
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = '3'
    et.SubElement(root, 'Row').text = str(ImgSize)
    et.SubElement(root, 'Col').text = str(ImgSize)
    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(meanImg, 'rows').text = '1'
    et.SubElement(meanImg, 'cols').text = str(ImgSize * ImgSize * 3)
    et.SubElement(meanImg, 'dt').text = 'f'
    et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(data, (ImgSize * ImgSize * 3))])

    tree = et.ElementTree(root)
    tree.write(fname)
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = '  '))

def saveTrainImages(filename, foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    data = {}
    dataMean = np.zeros((3, ImgSize, ImgSize)) # mean is in CHW format.
    with open('train_map.txt', 'w') as mapFile:
        with open('train_regrLabels.txt', 'w') as regrFile:
            for ifile in range(1, 6):
                with open(os.path.join('./cifar-10-batches-py', 'data_batch_' + str(ifile)), 'rb') as f:
                    if sys.version_info[0] < 3: 
                        data = cp.load(f)
                    else: 
                        data = cp.load(f, encoding='latin1')
                    for i in range(10000):
                        fname = os.path.join(os.path.abspath(foldername), ('%05d.png' % (i + (ifile - 1) * 10000)))
                        saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 4, mean=dataMean)
    dataMean = dataMean / (50 * 1000)
    saveMean('CIFAR-10_mean.xml', dataMean)

def saveTestImages(filename, foldername):
    if not os.path.exists(foldername):
      os.makedirs(foldername)
    with open('test_map.txt', 'w') as mapFile:
        with open('test_regrLabels.txt', 'w') as regrFile:
            with open(os.path.join('./cifar-10-batches-py', 'test_batch'), 'rb') as f:
                if sys.version_info[0] < 3: 
                    data = cp.load(f)
                else: 
                    data = cp.load(f, encoding='latin1')
                for i in range(10000):
                    fname = os.path.join(os.path.abspath(foldername), ('%05d.png' % i))
                    saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 0)
