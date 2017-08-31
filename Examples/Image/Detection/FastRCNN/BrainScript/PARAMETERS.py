from __future__ import print_function
import os
from imdb_data import imdb_data
import fastRCNN, time, datetime
from fastRCNN.pascal_voc import pascal_voc # as nmsPython
print (datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

dataset = "Grocery"
#dataset = "pascalVoc"
#dataset = "pascalVoc_aeroplanesOnly"
#dataset = "CustomDataset"

############################
# default parameters
############################
class Parameters():
    def __init__(self, datasetName):
        # cntk params
        self.datasetName = datasetName
        self.cntk_nrRois = 100      # how many ROIs to zero-pad. Use 100 to get quick result. Use 2000 to get good results.
        self.cntk_padWidth = 1000
        self.cntk_padHeight = 1000

        # directories
        self.rootDir = os.path.dirname(os.path.abspath(__file__))
        self.imgDir = os.path.join(self.rootDir, "..", "..", "..", "DataSets", datasetName)

        # derived directories
        self.procDir = os.path.join(self.rootDir, "proc", datasetName + "_{}".format(self.cntk_nrRois))
        self.resultsDir = os.path.join(self.rootDir, "results", datasetName + "_{}".format(self.cntk_nrRois))
        self.roiDir = os.path.join(self.procDir, "rois")
        self.cntkFilesDir = os.path.join(self.procDir, "cntkFiles")
        self.cntkTemplateDir = self.rootDir

        # ROI generation
        self.roi_minDimRel = 0.01      # minium relative width/height of a ROI
        self.roi_maxDimRel = 1.0       # maximum relative width/height of a ROI
        self.roi_minNrPixelsRel = 0    # minium relative area covered by ROI
        self.roi_maxNrPixelsRel = 1.0  # maximm relative area covered by ROI
        self.roi_maxAspectRatio = 4.0  # maximum aspect Ratio of a ROI vertically and horizontally
        self.roi_maxImgDim = 200       # image size used for ROI generation
        self.ss_scale = 100            # selective search ROIS: parameter controlling cluster size for segmentation
        self.ss_sigma = 1.2            # selective search ROIs: width of gaussian kernal for segmentation
        self.ss_minSize = 20           # selective search ROIs: minimum component size for segmentation
        self.grid_nrScales = 7         # uniform grid ROIs: number of iterations from largest possible ROI to smaller ROIs
        self.grid_aspectRatios = [1.0, 2.0, 0.5]    # uniform grid ROIs: aspect ratio of ROIs

        # thresholds
        self.train_posOverlapThres = 0.5 # threshold for marking ROIs as positive.
        self.nmsThreshold = 0.3          # Non-Maxima suppression threshold (in range [0,1]).
                                         # The lower the more ROIs will be combined. Used in 5_evaluateResults and 5_visualizeResults.

        self.cntk_num_train_images = -1          # set per data set below
        self.cntk_num_test_images = -1           # set per data set below
        self.cntk_mb_size = -1                   # set per data set below
        self.cntk_max_epochs = -1                # set per data set below
        self.cntk_momentum_time_constant = -1    # set per data set below

############################
# project-specific parameters
############################
class GroceryParameters(Parameters):
    def __init__(self, datasetName):
        super(GroceryParameters,self).__init__(datasetName)
        self.classes = ('__background__',  # always index 0
                   'avocado', 'orange', 'butter', 'champagne', 'eggBox', 'gerkin', 'joghurt', 'ketchup',
                   'orangeJuice', 'onion', 'pepper', 'tomato', 'water', 'milk', 'tabasco', 'mustard')

        # roi generation
        self.roi_minDimRel = 0.04
        self.roi_maxDimRel = 0.4
        self.roi_minNrPixelsRel = 2    * self.roi_minDimRel * self.roi_minDimRel
        self.roi_maxNrPixelsRel = 0.33 * self.roi_maxDimRel * self.roi_maxDimRel

        # model training / scoring
        self.classifier = 'nn'
        self.cntk_num_train_images = 25
        self.cntk_num_test_images = 5
        self.cntk_mb_size = 5
        self.cntk_max_epochs = 20
        self.cntk_momentum_time_constant = 10

        # postprocessing
        self.nmsThreshold = 0.01

        # database
        self.imdbs = dict()      # database provider of images and image annotations
        for image_set in ["train", "test"]:
            self.imdbs[image_set] = imdb_data(image_set, self.classes, self.cntk_nrRois, self.imgDir, self.roiDir, self.cntkFilesDir, boAddGroundTruthRois=(image_set!='test'))

class CustomDataset(Parameters):
    def __init__(self, datasetName):
        super(CustomDataset,self).__init__(datasetName)


class PascalParameters(Parameters):
    def __init__(self, datasetName):
        super(PascalParameters,self).__init__(datasetName)
        if datasetName.startswith("pascalVoc_aeroplanesOnly"):
            self.classes = ('__background__', 'aeroplane')
            self.lutImageSet = {"train": "trainval.aeroplaneOnly", "test": "test.aeroplaneOnly"}
        else:
            self.classes = ('__background__',  # always index 0
                       'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                       'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
            self.lutImageSet = {"train": "trainval", "test": "test"}

        # use cntk_nrRois = 4000. more than 99% of the test images have less than 4000 rois, but 50% more than 2000
        # model training / scoring
        self.classifier = 'nn'
        self.cntk_num_train_images = 5011
        self.cntk_num_test_images = 4952
        self.cntk_mb_size = 2
        self.cntk_max_epochs = 17
        self.cntk_momentum_time_constant = 20

        self.pascalDataDir = os.path.join(self.rootDir, "..", "..", "DataSets", "Pascal")
        self.imgDir = self.pascalDataDir

        # database
        self.imdbs = dict()
        for image_set, year in zip(["train", "test"], ["2007", "2007"]):
            self.imdbs[image_set] = fastRCNN.pascal_voc(self.lutImageSet[image_set], year, self.classes, self.cntk_nrRois, cacheDir=self.cntkFilesDir, devkit_path=self.pascalDataDir)
            print ("Number of {} images: {}".format(image_set, self.imdbs[image_set].num_images))

def get_parameters_for_dataset(datasetName=dataset):
    if datasetName == "Grocery":
        parameters = GroceryParameters(datasetName)
    elif datasetName.startswith("pascalVoc"):
        parameters = PascalParameters(datasetName)
    elif dataset.Name == "CustomDataset":
        parameters = CustomDataset(datasetName)
    else:
        ERROR

    ############################
    # computed parameters
    ############################
    nrClasses = len(parameters.classes)
    parameters.cntk_featureDimensions = {'nn': nrClasses}
    parameters.nrClasses = nrClasses

    assert parameters.cntk_padWidth == parameters.cntk_padHeight, "ERROR: different width and height for padding currently not supported."
    assert parameters.classifier.lower() in ['svm','nn'], "ERROR: only 'nn' or 'svm' classifier supported."
    assert not (parameters.datasetName == 'pascalVoc' and parameters.classifier == 'svm'), "ERROR: while technically possibly, writing 2nd-last layer of CNTK model for all pascalVOC images takes too much disk memory."

    print ("PARAMETERS: datasetName = " + datasetName)
    print ("PARAMETERS: cntk_nrRois = {}".format(parameters.cntk_nrRois))

    return parameters
