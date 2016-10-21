from __future__ import print_function
import os
import zipfile
import os
import os.path
try: 
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
    
if __name__ == "__main__":
    directory = "./../../DataSets"
    if not os.path.exists(directory + "/Grocery"):
        filename = directory + "/Grocery.zip"
        if not os.path.exists(filename):
            url = "https://www.cntk.ai/DataSets/Grocery/Grocery.zip"
            print ('Downloading data from ' + url + '...')
            urlretrieve(url, filename)
            try:
                print ('Extracting ' + filename + '...')
                with zipfile.ZipFile(filename) as myzip:
                    myzip.extractall(directory)
            finally:
                os.remove(filename)
            print ('Done.')
    else:
        print ('Data already available at ' + directory)
       
    directory = "./../../PretrainedModels"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + "/AlexNet.model"
    if not os.path.exists(filename):
        url = "https://www.cntk.ai/Models/AlexNet/AlexNet.model"
        print ('Downloading model from ' + url + '...')
        urlretrieve(url, filename)
        print ('Saved model as ' + filename)
    else:
        print ('CNTK model already available at ' + filename)
        
