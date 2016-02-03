import os
import sys
import struct
import cPickle as cp
from PIL import Image
import numpy as np
import xml.etree.cElementTree as et
import xml.dom.minidom

imgSize = 32

def saveImage(fname, data, label, mapFile, pad, **key_parms):
    # data in CIFAR-10 dataset is in CHW format.
    pixData = data.reshape((3, imgSize, imgSize))
    if ('mean' in key_parms):
        key_parms['mean'] += pixData

    if pad > 0:
        pixData = np.pad(pixData, ((0, 0), (pad, pad), (pad, pad)), mode = 'edge')

    img = Image.new('RGB', (imgSize + 2 * pad, imgSize + 2 * pad))
    pixels = img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            pixels[x, y] = (pixData[0][y][x], pixData[1][y][x], pixData[2][y][x])
    img.save(fname)
    mapFile.write("%s\t%d\n" % (fname, label))

def saveMean(fname, data):
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = '3'
    et.SubElement(root, 'Row').text = str(imgSize)
    et.SubElement(root, 'Col').text = str(imgSize)
    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(meanImg, 'rows').text = '1'
    et.SubElement(meanImg, 'cols').text = str(imgSize * imgSize * 3)
    et.SubElement(meanImg, 'dt').text = 'f'
    et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(data, (imgSize * imgSize * 3))])

    tree = et.ElementTree(root)
    tree.write(fname)
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = '  '))

if __name__ == "__main__":
    rootDir = r'C:\Data\CIFAR-10' + '\\'
    data = {}
    dataMean = np.zeros((3, imgSize, imgSize)) # mean is in CHW format.
    with open(rootDir + 'train_map.txt', 'w') as mapFile:
        for ifile in range(1, 6):
            with open(r'C:\Data\CIFAR-10\Python\data_batch_' + str(ifile), 'rb') as f:
                data = cp.load(f)
                for i in range(10000):
                    fname = '%sdata\\train\\%05d.png' % (rootDir, i + (ifile - 1) * 10000)
                    saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, 4, mean=dataMean)
    dataMean = dataMean / (50 * 1000)
    saveMean('%sdata\\CIFAR-10_mean.xml' % rootDir, dataMean)
    with open(rootDir + 'test_map.txt', 'w') as mapFile:
        with open(r'C:\Data\CIFAR-10\Python\test_batch', 'rb') as f:
            data = cp.load(f)
            for i in range(10000):
                fname = '%sdata\\test\\%05d.png' % (rootDir, i)
                saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, 0)
