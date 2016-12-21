from __future__ import print_function
import os
import os.path
import tarfile
import zipfile
try: 
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
    
def download_and_untar(url, filename, filesize):
    if not os.path.exists(filename):
        print ('Downloading ' + filesize + ' from ' + url + ', may take a while...')
        try:
            urlretrieve(url, filename)
        except (urllib.ContentTooShortError, IOError) as e:
            print ("Error downloading file: " + str(e))
            os.remove(filename)
            quit()
    else:
        print ('Found ' + filename)
    try:
        print ('Extracting ' + filename + '...')
        with tarfile.open(filename) as tar:
            tar.extractall()
        print ('Done.')
    finally:
        os.remove(filename)
    return;
    
if __name__ == "__main__":
    directory = "./VOCdevkit/VOC2007"
    if not os.path.exists(directory):
        download_and_untar(
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar", 
            "./VOCtrainval_06-Nov-2007.tar", 
            "450MB")
        download_and_untar(
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar", 
            "./VOCtest_06-Nov-2007.tar", 
            "430MB")
    else:
        print (directory + ' data already available.')
    
    directory = "./VOCdevkit/VOC2012"
    if not os.path.exists(directory):
        download_and_untar(
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            "./VOCtrainval_11-May-2012.tar", 
            "2GB")
    else:
        print (directory + ' data already available.')
               
    directory = "./selective_search_data"
    if not os.path.exists(directory):
        os.makedirs(directory)
        download_and_untar(
            "http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz", 
            "./selective_search_data.tgz", 
            "460MB")
    else:
        print (directory + ' data already available.')
    
