# Flowers Dataset

The Flowers dataset is a dataset for image classification created by the Visual Geometry Group at the University of Oxford. It consists of 102 different categories of flowers common to the UK. For more details see [[http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html]]

The Flowers dataset is not included in the CNTK distribution but can be easily
downloaded by cd to this directory, Examples/Image/DataSets/Flowers and running the following Python command:

`python install_flowers.py`

After running the script, you will see a 'jpg' folder that contains the images and three map files that split the roughly 8000 images into three sets of once 6000 and twice 1000 images.

The Flowers dataset is for example used in the Transfer Learning example, see [here](https://github.com/Microsoft/CNTK/wiki/Build-your-own-image-classifier-using-Transfer-Learning).
