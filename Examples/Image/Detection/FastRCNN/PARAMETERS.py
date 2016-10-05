from cntk_helpers import *
from imdb_data import imdb_data
import fastRCNN, time, datetime
from fastRCNN.pascal_voc import pascal_voc # as nmsPython
print datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

# dataset name
datasetName = "toy"
# datasetName = "pascalVoc"
# datasetName = "pascalVoc_aeroplanesOnly"
print "PARAMETERS: datasetName = " + datasetName


############################
# default parameters
############################
# directories
rootDir = "C:/src/CNTK/Examples/Image/Detection/FastRCNN/"
imgDir = rootDir + "data/" + datasetName + "/"
procDir = rootDir + "proc/" + datasetName + "/"
resultsDir = rootDir + "results/" + datasetName + "/"
roiDir = procDir + "rois/"
cntkFilesDir = procDir + "cntkFiles/"
cntkTemplateDir = rootDir

#cntk params
cntk_nrRois = 100  # how many ROIs to zero-pad
cntk_padWidth = 1000
cntk_padHeight = 1000
cntk_posOverlapThres = {"train": 0.5, "test": 0.5}  # only used for DNN training (as opposed to svm training)
cntk_featureDimensions = {'svm': 4096}

#training svm params
svm_C = 0.001
svm_B = 10.0            # intercept scaling
svm_nrEpochs = 5
svm_retrainLimit = 2000
svm_evictThreshold = -1.1
svm_posWeight = "auto" # "auto" # 2.0, else "balanced"
svm_targetNorm = 20.0      # Magic value from traditional R-CNN
svm_penality = 'l2'
svm_loss = 'l1'
svm_rngSeed = 3

#postprocessing
nmsThreshold = 0.3


############################
# project-specific parameters
############################
if datasetName.startswith("toy"):
    classes = ('__background__',  # always index 0
               "avocado", "orange", "butter", "champagne", "cheese", "eggBox", "gerkin", "joghurt", "ketchup",
               "orangeJuice", "onion", "pepper", "sausage", "tomato", "water", "apple", "milk",
               "tabasco", "soySauce", "mustard", "beer")

    # roi generation
    roi_minDimRel = 0.04
    roi_maxDimRel = 0.4
    roi_minNrPixelsRel = 0.06 * 0.06
    roi_maxNrPixelsRel = 0.33 * roi_maxDimRel * roi_maxDimRel
    roi_maxAspectRatio = 4.0
    roi_maxImgDim = 200
    ss_scale = 100
    ss_sigma = 1.2
    ss_minSize = 20
    grid_nrScales = 5
    grid_aspectRatios = [1.0, 2.0, 0.5]

    # model training / scoring
    classifier = 'nn'

    # database
    imdbs = dict()
    for image_set in ["train", "test"]:
        imdbs[image_set] = imdb_data(image_set, classes, cntk_nrRois, imgDir, roiDir, cntkFilesDir, boAddGroundTruthRois = (image_set!='test'))


elif datasetName.startswith("pascalVoc"):
    if datasetName.startswith("pascalVoc_aeroplanesOnly"):
        classes = ('__background__', 'aeroplane')
        lutImageSet = {"train": "trainval.aeroplaneOnly", "test": "test.aeroplaneOnly"}
    else:
        classes = ('__background__',  # always index 0
                   'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        lutImageSet = {"train": "trainval", "test": "test"}

    # model training / scoring
    classifier = 'nn'

    # database
    imdbs = dict()
    for image_set, year in zip(["train", "test"], ["2012", "2007"]):
        imdbs[image_set] = fastRCNN.pascal_voc(lutImageSet[image_set], year, classes, cntk_nrRois, cacheDir = cntkFilesDir)
        print "Number of {} images: {}".format(image_set, imdbs[image_set].num_images)

else:
     ERROR


############################
# computed parameters
############################
nrClasses = len(classes)
cntk_featureDimensions['nn'] = nrClasses

assert cntk_padWidth == cntk_padHeight, "ERROR: different width and height for padding currently not supported."
assert classifier.lower() in ['svm','nn'], "ERROR: only 'nn' or 'svm' classifier supported."
assert not (datasetName == 'pascalVoc' and classifier == 'svm'), "ERROR: while technically possibly, writing 2nd-last layer of CNTK model for all pascalVOC images takes too much disk memory."