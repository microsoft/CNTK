#Install:
- Install python 2.7.12 (64 bit required for Pascal Voc evaluation) https://www.python.org/downloads/windows/
- pip install -r requirements64.txt

#Setup:
- Run 'python install_fastrcnn.py' to download the example grocery data and the pretrained classification model.
- In PARAMETERS.py: Change 'rootdir' to the absolute path of the FastRCNN folder of your CNTK repository clone (only forward slashes, has to end with forward slash).

#Running Fast R-CNN on the provided grocery example data set:
- In PARAMETERS.py: make sure datasetName is set to "grocery".
- Run scripts A1, A2 and A3. All output will be written to a new subdirectory called 'proc'.
- Optionally run B scripts to visualize or evaluate after corresponding A step.



#Running Fast R-CNN on Pascal VOC data:
- Download the PAscal VOC data to <CntkRoot>/Examples/Image/Datasets/Pascal
-- 2007 trainval: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar 
-- 2007 test:     http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar 
-- 2012 trainval: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
- additionally you need selective_search_data: http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz
- in PARAMETERS.py set datasetName = "pascalVoc"


#Running on your own data
If you use another data set than the provided toy example or pascal:
- You can use scripts C1 and C2 to draw rectangles on new images and assign labels to those rectangles. The scripts will store the annotations in the correct format for CNTK Fast R-CNN.
- In PARAMETERS.py:
-- Pick a new name and assign it to 'datasetName'.
-- Adjust 'imgDir' to the directory where your images reside.
-- Adjust parameters under 'project-specific parameter' to your data, i.e. classes etc.
