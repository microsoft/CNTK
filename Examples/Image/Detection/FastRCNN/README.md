General:
- Install python 2.7.12
- Pip install -r requirements.txt

in PARAMETERS.py:
- Change 'rootdir' to the absolute path of the FastRCNN folder of your CNTK repository clone (only forward slashes, has to end with forward slash)
- for using Pascal VOC data see bottom of this file
- If you use another data set than the provided toy example or pascal
-- Pick a new name and assign it to 'datasetName'
-- Adjust 'imgDir' to the directory where your images reside
-- Adjust parameters under 'project-specific parameter' to your data, i.e. classes etc.

Running Fast R-CNN:
- Run scripts A1, A2 and A3
- 

Using Pascal data:
- you need both VOCdevkit2007 and VOCdevkit2012
- additionally you need selective_search_data
- in Params set pascalDataDir = pascal root dir ("C:/Temp/Pascal/")
