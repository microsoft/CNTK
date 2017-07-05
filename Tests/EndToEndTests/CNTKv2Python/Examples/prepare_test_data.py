# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import tarfile
import zipfile
from shutil import copyfile

envvar = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'

# Generic helper to read data from 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' to local machine
# One should use this helper when copying code is needed
# TODO: Update the other data loaders to reuse this code
def _data_copier(src_files, dst_files):
    src_files = [os.path.normpath(os.path.join((os.environ[envvar]), \
                    *src_file.split("/"))) for src_file in src_files]
                    
    dst_files = [os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), \
                    *dst_file.split("/"))) for dst_file in dst_files]                
    
    if not len(src_files) == len(dst_files):
        raise Exception('The length of src and dst should be same')
        
    for src_dst_file in zip(src_files, dst_files):
        # Note index 0 is the source and index 1 is destination
        if not os.path.isfile(src_dst_file[1]):
            # copy from backup location
            print("Copying file from: ", src_dst_file[0])
            print("Copying file to: ", src_dst_file[1])
            copyfile( src_dst_file[0], src_dst_file[1])
        else:
            print("Reusing cached file", src_dst_file[1])    

def prepare_CIFAR10_data(): 
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             *"../../../../Examples/Image/DataSets/CIFAR-10".split("/"))
    base_path = os.path.normpath(base_path)

    # If {train,test}_map.txt don't exist locally, copy to local location
    if not (os.path.isfile(os.path.join(base_path, 'train_map.txt')) and os.path.isfile(os.path.join(base_path, 'test_map.txt'))): 
        # copy from backup location 
        base_path_bak = os.path.join(os.environ[envvar],
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
                             *"../../../../Examples/Image/DataSets/ImageNet".split("/"))
    base_path = os.path.normpath(base_path)
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
    
    # If val1024_map.txt don't exist locally, copy to local location
    if not (os.path.isfile(os.path.join(base_path, 'train_map.txt')) and os.path.isfile(os.path.join(base_path, 'val_map.txt'))):
        # copy from backup location 
        base_path_bak = os.path.join(os.environ[envvar],
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

    if not os.path.isfile(os.path.join(base_path, 'test.txt')):
        # copy from backup location
        base_path_bak = os.path.join(os.environ[envvar],
                                     *"Image/Grocery".split("/"))
        base_path_bak = os.path.normpath(base_path_bak)

        zip_path = os.path.join(base_path, '..', 'Grocery.zip')
        copyfile(os.path.join(base_path_bak, 'Grocery.zip'), zip_path)
        with zipfile.ZipFile(zip_path) as myzip:
            myzip.extractall(os.path.join(base_path, '..'))

    return base_path


def prepare_fastrcnn_grocery_100_model():
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             *"../../../../Examples/Image/PretrainedModels".split("/"))
    base_path = os.path.normpath(base_path)

    if not os.path.isfile(os.path.join(base_path, 'Fast-RCNN_grocery100.model')):
        # copy from backup location
        base_path_bak = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                     *"PreTrainedModels/FRCN_Grocery/v0".split("/"))
        base_path_bak = os.path.normpath(base_path_bak)

        model_file_path = os.path.join(base_path, 'Fast-RCNN_grocery100.model')
        copyfile(os.path.join(base_path_bak, 'Fast-RCNN_grocery100.model'), model_file_path)
        print("copied model")
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
                    
# Read the flower and animal data set file
def prepare_resnet_v1_model():
    src_file = "PreTrainedModels/ResNet/v1/ResNet_18.model"
    dst_file = "Examples/Image/PretrainedModels/ResNet_18.model"
    
    _data_copier([src_file], [dst_file])
    
# Read the flower and animal data set file
def prepare_flower_data():
    src_files = ["Image/Flowers/102flowers.tgz", 
                 "Image/Flowers/imagelabels.mat", 
                 "Image/Flowers/imagelabels.mat"]

    dst_files = ["Examples/Image/DataSets/Flowers/102flowers.tgz", 
                 "Examples/Image/DataSets/Flowers/imagelabels.mat", 
                 "Examples/Image/DataSets/Flowers/imagelabels.mat"]  
    
    _data_copier(src_files, dst_files)
    
def prepare_animals_data():
    src_file = "Image/Animals/Animals.zip"
    dst_file = "Examples/Image/DataSets/Animals/Animals.zip"
    
    _data_copier([src_file], [dst_file])

def prepare_alexnet_v0_model():
    local_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         *"../../../../Examples/Image/PretrainedModels".split("/"))
    local_base_path = os.path.normpath(local_base_path)

    model_file = os.path.join(local_base_path, "AlexNet.model")

    if not os.path.isfile(model_file):
        external_model_path = os.path.join(os.environ[envvar], "PreTrainedModels", "AlexNet", "v0", "AlexNet.model")
        copyfile(external_model_path, model_file)
    return local_base_path

def prepare_UCF11_data():
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             *"../../../../Examples/Video/DataSets/UCF11".split("/"))
    base_path = os.path.normpath(base_path)
    if not os.path.isfile(os.path.join(base_path, 'test_map.csv')):
        # extract from backup location
        tar_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"DataSets/UCF11-v0.tar".split("/"))
        with tarfile.TarFile(tar_path) as mytar:
            mytar.extractall(base_path)

# TODO: We should have a common PTB (Penn Tree Bank) dataset for other tests also based on it
#       There is already similar data checked in Examples\SequenceToSequence\PennTreebank\Data
def prepare_WordLMWithSampledSoftmax_ptb_data():
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "Examples", "Text", "WordLMWithSampledSoftmax", "ptb")
    base_path = os.path.normpath(base_path)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    external_data_path = os.path.join(os.environ[envvar], "Text", "WordLMWithSampledSoftmax_ptb")
    src_files = ["test.txt", "token2freq.txt", "token2id.txt", "train.txt", "valid.txt", "vocab.txt", "freq.txt"]

    for src_file in src_files:
        if os.path.isfile(os.path.join(base_path, src_file)):
            continue
        copyfile(os.path.join(external_data_path, src_file), os.path.join(base_path, src_file))
