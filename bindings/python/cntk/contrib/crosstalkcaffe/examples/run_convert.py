# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
'''
Demo of examples

Args:
    network: The networks to be converted in Example, default converting all models
    log: The log file path
'''

import sys
import argparse
from cntk.contrib.crosstalkcaffe import CaffeConverter


NETWORK_DICT = {
    'AlexNet': 'Classification/AlexNet_ImageNet/global.json',
    'NIN': 'Classification/NIN_ImageNet/global.json',
    'VGG16': 'Classification/VGG_ImageNet/global_vgg16.json',
    'VGG19': 'Classification/VGG_ImageNet/global_vgg19.json',
    'GoogLeNet': 'Classification/GoogLeNet_ImageNet/global.json',
    'ResNet50': 'Classification/ResNet_ImageNet/global_resnet50.json',
    'ResNet101': 'Classification/ResNet_ImageNet/global_resnet101.json',
    'ResNet152': 'Classification/ResNet_ImageNet/global_resnet152.json',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', help='networks to convert, separated with comma',
                        required=False, default=None)
    parser.add_argument('-l', '--log', help='the path to redir the logs',
                        required=False, default='log.txt')
    args = vars(parser.parse_args())
    if args['log'] is not None:
        sys.stdout = open(args['log'], 'w')
    networks = NETWORK_DICT.keys() if args['network'] is None \
        else [item for item in args['network'].split(',')]
    for key in networks:
        CaffeConverter.from_model(NETWORK_DICT[key])
