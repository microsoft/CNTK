# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""
from __future__ import print_function
import os
from fastRCNN.test import test_net as evaluate_net
from fastRCNN.timer import Timer
from imdb_data import imdb_data
from cntk_helpers import makeDirectory, parseCntkOutput, DummyNet, deleteAllFilesInDirectory
import PARAMETERS


####################################
# Parameters
####################################
image_set = 'test' # 'train', 'test'

def evaluate_output():
    p = PARAMETERS.get_parameters_for_dataset()
    # parse cntk output
    print ("Parsing CNTK output for image set: " + image_set)
    cntkImgsListPath = os.path.join(p.cntkFilesDir, image_set + ".txt")
    outParsedDir = os.path.join(p.cntkFilesDir, image_set + "_parsed")
    cntkOutputPath = os.path.join(p.cntkFilesDir, image_set + ".z")

    # write cntk output for each image to separate file
    makeDirectory(outParsedDir)
    parseCntkOutput(cntkImgsListPath, cntkOutputPath, outParsedDir, p.cntk_nrRois, p.cntk_featureDimensions[p.classifier],
                    saveCompressed=True, skipCheck=True)

    # delete cntk output file which can be very large
    # deleteFile(cntkOutputPath)

    imdb = p.imdbs[image_set]
    net = DummyNet(4096, imdb.num_classes, outParsedDir)

    # create empty directory for evaluation files
    if type(imdb) == imdb_data:
        evalTempDir = None
    else:
        # pascal_voc implementation requires temporary directory for evaluation
        evalTempDir = os.path.join(p.procDir, "eval_mAP_" + image_set)
        makeDirectory(evalTempDir)
        deleteAllFilesInDirectory(evalTempDir, None)

    # compute mAPs
    evaluate_net(net, imdb, evalTempDir, None, p.classifier, p.nmsThreshold, boUsePythonImpl=True)

    print ("DONE.")
    return True

if __name__=='__main__':
    evaluate_output()
