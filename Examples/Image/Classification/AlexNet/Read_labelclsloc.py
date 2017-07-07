import os
import sys
import xml.etree.ElementTree as Et


def readlabel(xmlfilename):
    """
    Reads labels from the XML annotation file of validation set of
    Classification-localization challenge of ILSVRC dataset
    :param xmlfilename: string. Full path of the XML file.
    :return: List of labels in the XML file
    """
    assert os.path.exists(xmlfilename), "The file %s was not found." % (
        xmlfilename)
    root = Et.parse(xmlfilename).getroot()
    clsname = []
    for obj in root.findall('object'):
        for label in obj.findall('name'):
            clsname.append(label.text)
    return clsname
if __name__ == "__main__":
    name = readlabel(sys.argv[1])
    print(len(name))