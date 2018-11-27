from __future__ import print_function
import zipfile
import os
import sys
    
def extract_animal_data(source_zip_filepath, target_path):
    if not os.path.exists(os.path.join(target_path, "Animals", "Test")):
        print('Extracting ' + source_zip_filepath + ' to ' + target_path + '...')
        with zipfile.ZipFile(source_zip_filepath) as zip:
            zip.extractall(target_path) 
        print('Animal data extracted.')
    else:
        print('Animal data already available at ' + target_path + '/Animals')
    
extract_animal_data(sys.argv[1], sys.argv[2])
    