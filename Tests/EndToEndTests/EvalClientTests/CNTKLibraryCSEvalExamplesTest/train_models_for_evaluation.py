# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
import argparse
import zipfile

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ResNet", "Python"))
sys.path.append(os.path.join(abs_path, "..", "..", "CNTKv2Python", "Examples"))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "LanguageUnderstanding", "ATIS", "Python"))
import prepare_test_data
import TrainResNet_CIFAR10
import LanguageUnderstanding

def train_cifar_resnet_for_eval(test_device, output_dir):

    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    base_path = prepare_test_data.prepare_CIFAR10_data()

    # change dir to locate data.zip correctly
    os.chdir(base_path)

    # unzip test images for eval
    with zipfile.ZipFile(os.path.join(base_path, 'cifar-10-batches-py', 'data.zip')) as myzip:
        for fn in range(6):
            myzip.extract('data/train/%05d.png'%(fn), output_dir)
  
    if test_device == 'cpu':
        print('train cifar_resnet only on GPU device. Use pre-trained models.')
    else:
        print('training cifar_resnet on GPU device...')
        reader_train = TrainResNet_CIFAR10.create_reader(os.path.join(base_path, 'train_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), True)
        reader_test  = TrainResNet_CIFAR10.create_reader(os.path.join(base_path, 'test_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), False)
        TrainResNet_CIFAR10.train_and_evaluate(reader_train, reader_test, 'resnet20', epoch_size=512, max_epochs=1, profiler_dir=None, model_dir=output_dir)

    return base_path

def train_language_understanding_atis_for_eval(test_device, output_dir):

    abs_path   = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "LanguageUnderstanding", "ATIS", "Data"))
    reader = LanguageUnderstanding.create_reader(data_path + "/atis.train.ctf")
    model = LanguageUnderstanding.create_model()

    # train
    LanguageUnderstanding.train(reader, model, max_epochs=1, model_dir=output_dir)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test_device', help='the test device to run', required=True)

    args = vars(parser.parse_args())
    test_device = args['test_device']

    output_dir = os.path.dirname(os.path.abspath(__file__))
    print('the output_dir is {}.'.format(output_dir))
    print('the test_device is {}.'.format(test_device))
    train_language_understanding_atis_for_eval(test_device, output_dir)
    train_cifar_resnet_for_eval(test_device, output_dir)

