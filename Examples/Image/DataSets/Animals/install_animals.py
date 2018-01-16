# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import zipfile
import os
try:
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
    
def download_animals_data():
    base_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = os.path.join(base_folder, "..")
    if not os.path.exists(os.path.join(dataset_folder, "Animals", "Test")):
        filename = os.path.join(dataset_folder, "Animals.zip")
        if not os.path.exists(filename):
            url = "https://www.cntk.ai/DataSets/Animals/Animals.zip"
            print('Downloading data from ' + url + '...')
            urlretrieve(url, filename)
            
        try:
            print('Extracting ' + filename + '...')
            with zipfile.ZipFile(filename) as myzip:
                myzip.extractall(dataset_folder)
        finally:
            os.remove(filename)
        print('Done.')
    else:
        print('Data already available at ' + dataset_folder + '/Animals')
    
if __name__ == "__main__":
    download_animals_data()
    