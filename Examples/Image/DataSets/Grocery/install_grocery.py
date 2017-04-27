# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import zipfile
import os
from sys import platform
import shutil

try:
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
    
def download_grocery_data():
    base_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = os.path.join(base_folder, "..")
    if not os.path.exists(os.path.join(dataset_folder, "Grocery", "testImages")):
        filename = os.path.join(dataset_folder, "Grocery.zip")
        if not os.path.exists(filename):
            url = "https://www.cntk.ai/DataSets/Grocery/Grocery.zip"
            print('Downloading data from ' + url + '...')
            urlretrieve(url, filename)
            
        try:
            print('Extracting ' + filename + '...')
            with zipfile.ZipFile(filename) as myzip:
                myzip.extractall(dataset_folder)
            if platform != "win32":
                testfile  = os.path.join(dataset_folder, "grocery", "test.txt")
                unixfile = os.path.join(dataset_folder, "grocery", "test_unix.txt")
                out = open(unixfile, 'w')
                with open(testfile) as f:
                    for line in f:
                        out.write(line.replace('\\', '/'))
                out.close()
                shutil.move(unixfile, testfile)
        finally:
            os.remove(filename)
        print('Done.')
    else:
        print('Data already available at ' + dataset_folder + '/Grocery')
    
if __name__ == "__main__":
    download_grocery_data()
    
