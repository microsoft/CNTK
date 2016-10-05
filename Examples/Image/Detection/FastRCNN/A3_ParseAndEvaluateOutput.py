#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""
import importlib
from fastRCNN.test import test_net
from fastRCNN.timer import Timer
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
image_set = 'test' # 'train', 'test'


####################################
# Main
####################################
# parse cntk output
print "Parsing CNTK output for image set: " + image_set
cntkImgsListPath = cntkFilesDir + image_set + ".txt"
outParsedDir = cntkFilesDir + image_set + "_parsed/"
cntkOutputPath = cntkFilesDir + image_set + ".z"

# write cntk output for each image to separate file
makeDirectory(outParsedDir)
parseCntkOutput(cntkImgsListPath, cntkOutputPath, outParsedDir, cntk_nrRois, cntk_featureDimensions[classifier],
                saveCompressed = True, skipCheck = True)

# delete cntk output file which can be very large
# deleteFile(cntkOutputPath)

imdb = imdbs[image_set]
net = DummyNet(4096, imdb.num_classes, outParsedDir)

# create empty directory for evaluation files
if type(imdb) == imdb_data:
    evalTempDir = None
else:
    # pascal_voc implementation requires temporary directory for evaluation
    evalTempDir = os.path.join(procDir, "eval_mAP_" + image_set)
    makeDirectory(evalTempDir)
    deleteAllFilesInDirectory(evalTempDir, None)

# compute mAPs
test_net(net, imdb, evalTempDir, None, classifier, nmsThreshold, boUsePythonImpl = True)

print "DONE."