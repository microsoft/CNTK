from __future__ import print_function
from cntk_helpers import *
from imdb_data import imdb_data
import fastRCNN, time, datetime
from fastRCNN.pascal_voc import pascal_voc # as nmsPython
print (datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

# dataset name
datasetName = "grocery"
# datasetName = "pascalVoc"
# datasetName = "pascalVoc_aeroplanesOnly"


############################
# default parameters
############################
# cntk params
cntk_nrRois = 100      # how many ROIs to zero-pad. Use 100 to get quick result. Use 2000 to get good results.
cntk_padWidth = 1000
cntk_padHeight = 1000

# directories
rootDir = os.path.dirname(os.path.realpath(sys.argv[0])) + "/"
imgDir = os.path.join(rootDir, "../../DataSets/" + datasetName+ "/")
pascalDataDir = os.path.join(rootDir, "../../DataSets/Pascal/")

# derived directories
procDir = os.path.join(rootDir, "proc/" + datasetName + "_{}/".format(cntk_nrRois))
resultsDir = os.path.join(rootDir, "results/" + datasetName + "_{}/".format(cntk_nrRois))
roiDir = os.path.join(procDir, "rois/")
cntkFilesDir = os.path.join(procDir, "cntkFiles/")
cntkTemplateDir = rootDir

# ROI generation
roi_minDimRel = 0.01      # minium relative width/height of a ROI
roi_maxDimRel = 1.0       # maximum relative width/height of a ROI
roi_minNrPixelsRel = 0    # minium relative area covered by ROI
roi_maxNrPixelsRel = 1.0  # maximm relative area covered by ROI
roi_maxAspectRatio = 4.0  # maximum aspect Ratio of a ROI vertically and horizontally
roi_maxImgDim = 200       # image size used for ROI generation
ss_scale = 100            # selective search ROIS: parameter controlling cluster size for segmentation
ss_sigma = 1.2            # selective search ROIs: width of gaussian kernal for segmentation
ss_minSize = 20           # selective search ROIs: minimum component size for segmentation
grid_nrScales = 7         # uniform grid ROIs: number of iterations from largest possible ROI to smaller ROIs
grid_aspectRatios = [1.0, 2.0, 0.5]    # uniform grid ROIs: aspect ratio of ROIs

# thresholds
train_posOverlapThres = 0.5 # threshold for marking ROIs as positive.
nmsThreshold = 0.3          # Non-Maxima suppression threshold (in range [0,1]).
                            # The lower the more ROIs will be combined. Used in 5_evaluateResults and 5_visualizeResults.

cntk_num_train_images = -1          # set per data set below
cntk_num_test_images = -1           # set per data set below
cntk_mb_size = -1                   # set per data set below
cntk_max_epochs = -1                # set per data set below
cntk_momentum_time_constant = -1    # set per data set below

############################
# project-specific parameters
############################
if datasetName.startswith("grocery"):
    classes = ('__background__',  # always index 0
               'avocado', 'orange', 'butter', 'champagne', 'eggBox', 'gerkin', 'joghurt', 'ketchup',
               'orangeJuice', 'onion', 'pepper', 'tomato', 'water', 'milk', 'tabasco', 'mustard')

    # roi generation
    roi_minDimRel = 0.04
    roi_maxDimRel = 0.4
    roi_minNrPixelsRel = 2    * roi_minDimRel * roi_minDimRel
    roi_maxNrPixelsRel = 0.33 * roi_maxDimRel * roi_maxDimRel

    # model training / scoring
    classifier = 'nn'
    cntk_num_train_images = 25
    cntk_num_test_images = 5
    cntk_mb_size = 5
    cntk_max_epochs = 20
    cntk_momentum_time_constant = 10

    # postprocessing
    nmsThreshold = 0.01

    # database
    imdbs = dict()      # database provider of images and image annotations
    for image_set in ["train", "test"]:
        imdbs[image_set] = imdb_data(image_set, classes, cntk_nrRois, imgDir, roiDir, cntkFilesDir, boAddGroundTruthRois = (image_set!='test'))


elif datasetName.startswith("pascalVoc"):
    imgDir = pascalDataDir
    if datasetName.startswith("pascalVoc_aeroplanesOnly"):
        classes = ('__background__', 'aeroplane')
        lutImageSet = {"train": "trainval.aeroplaneOnly", "test": "test.aeroplaneOnly"}
    else:
        classes = ('__background__',  # always index 0
                   'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        lutImageSet = {"train": "trainval", "test": "test"}

    # use cntk_nrRois = 4000. more than 99% of the test images have less than 4000 rois, but 50% more than 2000
    # model training / scoring
    classifier = 'nn'
    cntk_num_train_images = 5011
    cntk_num_test_images = 4952
    cntk_mb_size = 2
    cntk_max_epochs = 17
    cntk_momentum_time_constant = 20

    # database
    imdbs = dict()
    for image_set, year in zip(["train", "test"], ["2007", "2007"]):
        imdbs[image_set] = fastRCNN.pascal_voc(lutImageSet[image_set], year, classes, cntk_nrRois, cacheDir = cntkFilesDir, devkit_path=pascalDataDir)
        print ("Number of {} images: {}".format(image_set, imdbs[image_set].num_images))

else:
     ERROR


############################
# computed parameters
############################
nrClasses = len(classes)
cntk_featureDimensions = {'nn': nrClasses}

assert cntk_padWidth == cntk_padHeight, "ERROR: different width and height for padding currently not supported."
assert classifier.lower() in ['svm','nn'], "ERROR: only 'nn' or 'svm' classifier supported."
assert not (datasetName == 'pascalVoc' and classifier == 'svm'), "ERROR: while technically possibly, writing 2nd-last layer of CNTK model for all pascalVOC images takes too much disk memory."

print ("PARAMETERS: datasetName = " + datasetName)
print ("PARAMETERS: cntk_nrRois = {}".format(cntk_nrRois))
