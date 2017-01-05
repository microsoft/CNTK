# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import zipfile
import os
import os.path
try: 
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
    
if __name__ == "__main__":
    base_folder = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_folder, "..", "..", "DataSets")
    if not os.path.exists(os.path.join(directory, "grocery", "testImages")):
        filename = os.path.join(directory, "Grocery.zip")
        if not os.path.exists(filename):
            url = "https://www.cntk.ai/DataSets/Grocery/Grocery.zip"
            print('Downloading data from ' + url + '...')
            urlretrieve(url, filename)
            
        try:
            print('Extracting ' + filename + '...')
            with zipfile.ZipFile(filename) as myzip:
                myzip.extractall(directory)
        finally:
            os.remove(filename)
        print('Done.')
    else:
        print('Data already available at ' + directory + '/grocery')
        
    directory = os.path.join(base_folder, "..", "..", "PretrainedModels")
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, "AlexNet.model")
    if not os.path.exists(filename):
        url = "https://www.cntk.ai/Models/AlexNet/AlexNet.model"
        print('Downloading model from ' + url + ', may take a while...')
        urlretrieve(url, filename)
        print('Saved model as ' + filename)
    else:
        print('CNTK model already available at ' + filename)
        
