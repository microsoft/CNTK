General:
- Install python 2.7.12
- Pip install -r requirements.txt

in PARAMETERS.py:
- Change 'rootdir' to the absolute path of the FastRCNN folder of your CNTK repository clone (only forward slashes, has to end with forward slash)
- If you use another data set than the provided toy example
-- Pick a new name and assign it to 'datasetName'
-- Adjust 'imgDir' to the directory where your images reside
-- Adjust parameters under 'project-specific parameter' to your data, i.e. classes etc.
