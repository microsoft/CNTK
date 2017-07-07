import sys
import time
from Read_labelclsloc import *

def readtrain(traindir):
    """
    Reads the training subset of CLS-LOC challenge of ILSVRC 2015
    :param traindir: The folder containing the training class subfolders
    :returns: traindict(A dictionary with full path to each image as a key, and the
     corresponding label name as the corresponding value.).
     classdict(A dictionary with label names as key and corresponding number as value)
    """
    categories = []
    timeinit = time.time()
    trainimgs = []
    imglbl = []
    for entry in os.scandir(traindir):
        if entry.is_dir():
            categories.append(entry.name)
            for files in os.scandir(os.path.join(traindir, entry.name)):
                if files.is_file() and files.name.endswith('.JPEG'):
                    trainimgs.append(os.path.join(traindir, entry.name,
                                                  files.name))
                    imglbl.append(entry.name)
    classdict = dict(zip(categories, range(len(categories))))
    imglbl = list(map(lambda x: classdict[x], imglbl))
    traindict = dict(zip(trainimgs, imglbl))
    print("""Time taken to identify the training images and prepare the class
    label dictionary = %.2f seconds""" % (time.time() - timeinit))
    return traindict, classdict


def readval(valdir, anndir, classdict):
    """
    Reads the validation images of the CLS-LOC challenge of the ILSVRC 2015
    :param valdir: the folder containing the validation subset images.
    :param anndir: The folder containing the annotation files of validation data
    :param classdict: Dictionary mapping ILSVRC training labels to positive
    integers, computed using `ReadTrain()`
    :return: A dictionary. Keys are the full paths to the validation images.
    The values are the corresponding labels.
    """

    valdict = {}
    timeinit = time.time()
    for entry in os.scandir(valdir):
        valfile = os.path.join(valdir, entry.name)
        annfile = os.path.join(anndir,
                               os.path.splitext(os.path.basename(entry.name))[
                                   0]
                               + '.xml')
        labels = readlabel(annfile)
        labels = list(set(labels))
        valdict[valfile] = classdict[labels[0]]
        """
        if len(lbl) > 1:
            print('Multiple labels found.')
            print(lbl)
        """
    print("""Time taken to identify the labels for validation image = %.2f
         seconds""" % (time.time() - timeinit))
    return valdict


if __name__ == "__main__":
    traindict, classdict = readtrain(sys.argv[1])
    valdict = readval(sys.argv[2], sys.argv[3], classdict)
    #print(traindict)
    #print(classdict)
    #print(valdict)
    with  open("train_map.txt", "w") as f:
        for  key in traindict.keys():
            f.write(key + "\t" + str(traindict[key]) + "\n")

    with  open("val_map.txt", "w") as f:
        for  key in valdict.keys():
            f.write(key + "\t" + str(valdict[key]) + "\n")

    classdict = {v: k for k, v in classdict.items()}

    with  open("classmappings.txt", "w") as f:
        for  key in classdict.keys():
            f.write(str(key) + "\t" + classdict[key] + "\n")

            
