Install:
- Install python 2.7.12 (64 bit required for Pascal Voc evaluation) https://www.python.org/downloads/windows/
- pip install -r requirements64.txt

Setup:
- Download the data and the pretrained classification model to the corresponding folders (AlexNet.89 currently)
- In PARAMETERS.py: Change 'rootdir' to the absolute path of the FastRCNN folder of your CNTK repository clone (only forward slashes, has to end with forward slash).

Running Fast R-CNN on the provided toy data set:
- In PARAMETERS.py: make sure datasetName is set to "toy".
- Run scripts A1, A2 and A3. All output will be written to a new subdirectory called 'proc'.
- Optionally run B scripts to visualize or evaluate after corresponding A step.


Running Fast R-CNN on Pascal VOC data:
- you need both VOCdevkit2007 and VOCdevkit2012
- additionally you need selective_search_data
- in Params set pascalDataDir = pascal root dir ("C:/Temp/Pascal/")
- in PARAMETERS.py set datasetName = "pascalVoc"

If you use another data set than the provided toy example or pascal:
- You can use scripts C1 and C2 to draw rectangles on new images and assign labels to those rectangles. The scripts will store the annotations in the correct format for CNTK Fast R-CNN.
- In PARAMETERS.py:
-- Pick a new name and assign it to 'datasetName'.
-- Adjust 'imgDir' to the directory where your images reside.
-- Adjust parameters under 'project-specific parameter' to your data, i.e. classes etc.
