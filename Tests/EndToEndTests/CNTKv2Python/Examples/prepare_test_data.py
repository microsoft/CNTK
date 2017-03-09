# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import zipfile
from shutil import copyfile

def prepare_CIFAR10_data(): 
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             *"../../../../Examples/Image/DataSets/CIFAR-10".split("/"))
    base_path = os.path.normpath(base_path)

    # If {train,test}_map.txt don't exist locally, copy to local location
    if not (os.path.isfile(os.path.join(base_path, 'train_map.txt')) and os.path.isfile(os.path.join(base_path, 'test_map.txt'))): 
        # copy from backup location 
        base_path_bak = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                     *"Image/CIFAR/v0/cifar-10-batches-py".split("/"))
        base_path_bak = os.path.normpath(base_path_bak)
        
        copyfile(os.path.join(base_path_bak, 'train_map.txt'), os.path.join(base_path, 'train_map.txt'))
        copyfile(os.path.join(base_path_bak, 'test_map.txt'), os.path.join(base_path, 'test_map.txt'))
        if not os.path.isdir(os.path.join(base_path, 'cifar-10-batches-py')): 
            os.mkdir(os.path.join(base_path, 'cifar-10-batches-py'))
        copyfile(os.path.join(base_path_bak, 'data.zip'), os.path.join(base_path, 'cifar-10-batches-py', 'data.zip'))
        copyfile(os.path.join(base_path_bak, 'CIFAR-10_mean.xml'), os.path.join(base_path, 'CIFAR-10_mean.xml'))
    return base_path

def prepare_ImageNet_data():
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             *"../../../../Examples/Image/DataSets/ImageNet/test_data".split("/"))
    base_path = os.path.normpath(base_path)
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
    
    # If val1024_map.txt don't exist locally, copy to local location
    if not (os.path.isfile(os.path.join(base_path, 'train_map.txt')) and os.path.isfile(os.path.join(base_path, 'val_map.txt'))):
        # copy from backup location 
        base_path_bak = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                     *"Image/ImageNet/2012/v0".split("/"))
        base_path_bak = os.path.normpath(base_path_bak)
        
        copyfile(os.path.join(base_path_bak, 'val1024_map.txt'), os.path.join(base_path, 'train_map.txt'))
        copyfile(os.path.join(base_path_bak, 'val1024_map.txt'), os.path.join(base_path, 'val_map.txt'))
        copyfile(os.path.join(base_path_bak, 'val1024.zip'), os.path.join(base_path, 'val1024.zip'))
    return base_path

def prepare_Grocery_data():
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             *"../../../../Examples/Image/DataSets/Grocery".split("/"))
    base_path = os.path.normpath(base_path)

    # If val1024_map.txt don't exist locally, copy to local location
    if not os.path.isfile(os.path.join(base_path, 'test.txt')):
        # copy from backup location
        base_path_bak = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                     *"Image/Grocery".split("/"))
        base_path_bak = os.path.normpath(base_path_bak)

        zip_path = os.path.join(base_path, '..', 'Grocery.zip')
        copyfile(os.path.join(base_path_bak, 'Grocery.zip'), zip_path)
        with zipfile.ZipFile(zip_path) as myzip:
            myzip.extractall(os.path.join(base_path, '..'))

    return base_path

def an4_dataset_directory():
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             *"../../../../Examples/Speech/AN4/Data".split("/"))

    base_path = os.path.normpath(base_path)
    return base_path

def cmudict_dataset_directory():
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             *"../../../../Examples/SequenceToSequence/CMUDict/Data".split("/"))

    base_path = os.path.normpath(base_path)
    return base_path